from abc import ABC

import numpy as np
import torch.utils.data as data
import cv2
from typing import Dict, List
# from path import Path
import random
import os
import glob
from scipy import ndimage
import json
from multiprocessing import Pool
from tqdm import tqdm


# from collections import defaultdict


# a simple demo to count scene key frames
# 步骤：
# 1. 每个object出一个key frame fusion（按照Unprojection的方式融合）
# 2. 按原来的方法存

def bbox_filter(bbox, delta_px=10):
    """
    |x1 - x2| or |y1 - y2| should be greater than `delta_px`
    """
    return abs(bbox[0] - bbox[2]) > delta_px and abs(bbox[1] - bbox[3]) > delta_px


class CamScannetScene(ABC):
    def __init__(self, scene_root='/data/ScanNet/uncompressed/', scans_root='/data/ScanNet/scans/',
                 scene_name='scene0000_00', dump_root='',
                 args=None, debug=True):
        self.args = args

        self.scene_name = scene_name
        self.scene_root = os.path.join(scene_root, scene_name)
        self.scans_root = os.path.join(scans_root, scene_name)
        self.dump_root = dump_root

        self.meta_2d_info = self.read_meta(os.path.join(self.scene_root, '_info.txt'))
        with open(os.path.join(self.scans_root, '%s.aggregation.json' % self.scene_name), 'r') as f:
            self.meta_3d_info = json.load(f)

        self.max_instance_count = len(self.meta_3d_info['segGroups'])

        self.frame_list = list(sorted(glob.glob(os.path.join(self.scene_root, 'frame-*.color.jpg'))))
        self.pose_list = [k.replace('color.jpg', 'pose.txt') for k in self.frame_list]
        self.check_integrity()
        self.frame2instance_dict = self.get_frame2instance_exist_dict()
        self.instance2frame_range_dict = self.get_instance_frame_list()  # {0: [0, 2, 5, 8, ...], 1:...}

        self.instance2kf_dict = {}

        self.dump_ids_kf_bbox()
        # if debug:

    def dump_ids_kf_bbox(self):  # 应该按key frame group by!!!
        dump_dict = {
            'scene_name': self.scene_name,
            'max_instance_count': self.max_instance_count,
            'key_frames': []
        }
        kf_set = set()
        for instance_id in range(self.max_instance_count):
            kf_list = self.compute_key_frame(self.instance2frame_range_dict[instance_id])
            # dump_dict['instances'].append({
            #     'instance_id': instance_id,
            #     'key_frame': kf_list
            # })
            kf_set.update(kf_list)
            # self.instance2kf_dict[instance_id] = kf_list

        # we only compute bbox for all key frames to reduce computation
        for kf_id in kf_set:
            instance_img = cv2.imread(os.path.join(self.scans_root, 'instance-filt', '%d.png' % kf_id),
                                      cv2.IMREAD_UNCHANGED)  # [968, 1296]
            kf_bbox = []
            kf_ids = []
            for instance_id in np.unique(instance_img):
                s = ndimage.generate_binary_structure(2, 2)  # 8-way connect, instead of the default 4-way
                labeled_array, _ = ndimage.label(np.array(instance_img == instance_id, dtype=int),
                                                 s)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
                largest_cc_mask_xy = np.where(labeled_array == (np.bincount(labeled_array.flat)[1:].argmax() + 1))
                x_min, x_max = largest_cc_mask_xy[1].min(), largest_cc_mask_xy[1].max()
                y_min, y_max = largest_cc_mask_xy[0].min(), largest_cc_mask_xy[0].max()
                # 2021年11月25日: add bbox filter
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                if bbox_filter(bbox):
                    kf_bbox.append(bbox)
                    kf_ids.append(int(instance_id))

            dump_dict['key_frames'].append({
                'key_frame_id': kf_id,
                'id': kf_ids,
                'bbox': kf_bbox
            })

        with open(os.path.join(self.dump_root, '%s.kf.bbox.json' % self.scene_name), 'w') as f:
            json.dump(dump_dict, f, indent=2)

    def check_integrity(self):
        # frame length check
        assert int(self.meta_2d_info['m_frames.size']) == len(self.frame_list)

        # self.frame_list = self.frame_list[:100]

    def get_frame2instance_exist_dict(self):  # get <frame_id: ins_id1, ins_id2, ...>
        # usage: whether obj x in frame y: `x in res[y]`
        res = {}
        for frame_id in range(len(self.frame_list)):
            instance_img = cv2.imread(os.path.join(self.scans_root, 'instance-filt', '%d.png' % frame_id),
                                      cv2.IMREAD_UNCHANGED)  # [968, 1296]
            unique_ids = np.unique(instance_img)
            res[frame_id] = set(unique_ids)
        return res

    def get_instance_frame_list(self):
        res = {instance_id: [] for instance_id in range(self.max_instance_count)}
        for instance_id in range(self.max_instance_count):
            for frame_id in range(len(self.frame_list)):
                if instance_id in self.frame2instance_dict[frame_id]:
                    res[instance_id].append(frame_id)  # make sure frame_list keep ascending

        return res

    def compute_key_frame(self, range_list=None):
        """
        compute key frame given a frame segment <l, r>
        """
        if range_list is None:
            range_list = range(0, len(self.frame_list))

        key_frame_ids = []

        last_pose = None
        for frame_id in range_list:
            cam_pose = np.loadtxt(self.pose_list[frame_id]).astype(np.float32)
            if last_pose is None:
                last_pose = cam_pose
                # the first frame always a key frame
                key_frame_ids.append(frame_id)
            else:  # check if current frame is a key frame
                angle = np.arccos(
                    ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                        [0, 0, 1])).sum())
                dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])  # norm across the translation

                if angle > (15 / 180) * np.pi or dis > 0.1:  # 15 degree or 0.1m
                    # if angle > (15 / 180) * np.pi:
                    #     print('trigger angle at Frame#{}'.format(frame_id))
                    # if dis > 0.1:
                    #     print('trigger dis at Frame#{}'.format(frame_id))

                    last_pose = cam_pose
                    key_frame_ids.append(frame_id)

        return key_frame_ids

    @staticmethod
    def read_meta(txt_path):
        res = {}
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                k, v = line.split('=')
                k, v = k.strip(), v.strip()
                res[k] = v

        return res


def submit_job(scene_name):
    _ = CamScannetScene(scene_name=scene_name, dump_root='/data/meta-ScanNet/key_frame_bbox/')
    return None


if __name__ == '__main__':
    # 首先做所有frames的key frame selection
    # dataset = CamScannetScene('/data/ScanNet/uncompressed/', '/data/ScanNet/scans/', 'scene0000_00')
    # key_frame_ids_for_obj_4 = dataset.compute_key_frame(dataset.instance2frame_range_dict[4])
    # print("instance 4 exists in frame list {}, where key frames are {}.".format(dataset.instance2frame_range_dict[4],
    #                                                                             key_frame_ids_for_obj_4))
    # 尝试对每个instance做key frame selection
    scene_list = os.listdir('/data/ScanNet/scans/')
    with Pool(12) as p:
        r = list(tqdm(p.imap(submit_job, scene_list), total=len(scene_list)))
