"""
clip_add usage:
export CUDA_VISIBLE_DEVICES=1; export PYTHONPATH=~/CLIP-Transfer; python3 export_bbox_feat.py --extractor=clip_add --train_list=/data/meta-ScanNet/used_train_scene.txt
"""
import os
from models.offline_extractor.rcnn_pretrained import RCNNExtractor
from models.offline_extractor.clip_pretrained import CLIPExtractor
import argparse
import glob
from utils import prepare_cuda
from tqdm import tqdm
import json
from PIL import Image
from typing import List
import numpy as np

FACTORY = {
    'clip': CLIPExtractor,
    'clip_add': CLIPExtractor,
    'rcnn': RCNNExtractor
}

FRAME_PATH = '/data/ScanNet/uncompressed/{scene_name}/frame-{frame_id:0>6d}.color.jpg'


def read_lines(file_name):
    trimmed_lines = []
    with open(file_name) as fin:
        for line in fin:
            trimmed_lines.append(line.rstrip())
    return trimmed_lines


#
# kept_prefix = read_lines('/data/meta-ScanNet/used_train_scene.txt')
# # kept_set = set()
#
# all_file_list = os.listdir('/dev/shm/clip')
#
# for file in tqdm(all_file_list):
#     for i in range(len(kept_prefix)):  # scene0000_00
#         if kept_prefix[i] in file:
#             break  # kept
#         if i == len(kept_prefix) - 1:
#             os.remove('/dev/shm/clip/' + file)  # delete


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_path', type=str, default='/data/meta-ScanNet/key_frame_bbox/')
    parser.add_argument('--extractor', type=str, default='rcnn', choices=['clip', 'rcnn', 'clip_add'])
    parser.add_argument('--train_list', type=str, default=None)
    # parser.add_argument('--bbox_invalid_px', type=int, default=10)

    ### Options for RCNN
    parser.add_argument(
        "--detection_cfg", type=str,
        default='/data/pretrained_models/detectron_model.yaml',
        help="Detectron config file; download it from https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    )
    parser.add_argument(
        "--detection_model", type=str,
        default='/data/pretrained_models/detectron_model.pth',
        help="Detectron model file; download it from https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    )

    ### Option for CLIP
    parser.add_argument('--clip_type', type=str, default='RN50x16')

    parser.add_argument('--output_path', type=str, default='/data/meta-ScanNet/key_frame_feature/')

    return parser.parse_args()


def get_scaled_bbox(bbox, img_shape) -> List[int]:
    x1, y1, x2, y2 = bbox
    # x1, y1, x2, y2 = max(0, (2 * x1 - x2)), max(0, (2 * y1 - y2)), min(img_shape[2], (2 * x2 - x1)), min(
    #     img_shape[1], (2 * y2 - y1))  # opencv-style
    x1, y1, x2, y2 = max(0, (2 * x1 - x2)), max(0, (2 * y1 - y2)), min(img_shape[0], (2 * x2 - x1)), min(
        img_shape[1], (2 * y2 - y1))  # PIL style
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    args = parse_args()
    args.output_path = os.path.join(args.output_path, args.extractor)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    prepare_cuda()
    extractor = FACTORY[args.extractor](args)

    pbar = list(glob.glob(os.path.join(args.bbox_path, '*.kf.bbox.json')))

    if args.train_list is not None:
        scene_list = read_lines(args.train_list)
        pbar = [k for k in pbar if any(scene in k for scene in scene_list)]  # only kept trained scene

    pbar = tqdm(pbar)

    for scene_kf in pbar:
        with open(scene_kf, 'r') as f:
            scene_kf = json.load(f)

        scene_name = scene_kf['scene_name']

        for kf in scene_kf['key_frames']:  # for each key frame
            instance_ids: List[int] = kf['id']
            bboxes: List[List[int]] = kf['bbox']
            frame_id: int = kf['key_frame_id']

            # instance_ids, bboxes = [], []
            # # filter out invalid box
            # for ins_id, bbox in zip(kf['id'], kf['bbox']):
            #     if bbox_filter(bbox, args.bbox_invalid_px):
            #         instance_ids.append(ins_id)
            #         bboxes.append(bbox)

            # construct key frame path
            # if args.extractor == 'clip':
            #     # kf_img = Image.open(FRAME_PATH.format(scene_name=scene_name, frame_id=frame_id))
            #     kf_img = cv2.imread(FRAME_PATH.format(scene_name=scene_name, frame_id=frame_id))
            # else:
            kf_img = Image.open(FRAME_PATH.format(scene_name=scene_name, frame_id=frame_id))

            if args.extractor == 'clip_add':
                # bbox gets scaled up
                bboxes = [get_scaled_bbox(bbox, kf_img.size) for bbox in bboxes]
                # load old feat
                old_feat = np.load(f'/dev/shm'
                                   f'/clip/{scene_name}_kf_{frame_id}.npy', allow_pickle=True).item()['feat']

            bboxes = np.array(bboxes, dtype=np.float32)
            kf_bbox_feat = extractor.get_bbox_feature(kf_img, bboxes)  # [bbox_num, 2048/768]

            if args.extractor == 'clip_add':
                kf_bbox_feat += old_feat

            # save scene_name + key_frame to npy file
            output_path = os.path.join(args.output_path, f'{scene_name}_kf_{frame_id}.npy')
            np.save(output_path, {
                'scene_name': scene_name,
                'key_frame_id': frame_id,
                'instance_id': instance_ids,
                'bbox': bboxes,  # numpy obj: [bbox_num, 4]
                'feat': kf_bbox_feat  # # numpy obj: [bbox_num, feat_size]
            })
