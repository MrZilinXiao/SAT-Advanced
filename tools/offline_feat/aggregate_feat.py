import os
import glob
import json
import numpy as np
from tqdm import tqdm

INPUT_DIR = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat'
DEPTH_DIR = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat_depth'
# OUTPUT_DIR='/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat_aggregate'
OUTPUT_DIR = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat_aggregate_frameid'
INFO_DIR = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k'
ANNO_DIR = '/localdisk2/DATASET/scannet/scans_train'

scene_list = [x.split('/')[-1] for x in glob.glob('%s/*' % INPUT_DIR)]

for scene_name in tqdm(scene_list):
    save_path = os.path.join(OUTPUT_DIR, '%s.npy' % scene_name)
    cam_pose_path = os.path.join(INFO_DIR, scene_name, 'pose')
    obj_feat_path = os.path.join(INPUT_DIR, scene_name)
    depth_feat_path = os.path.join(DEPTH_DIR, scene_name)
    instance_idlist_frames = []
    if os.path.isfile(save_path):
        continue

    for sampled_id in range(0, int(1e6), 100):
        obj_info_file = os.path.join(obj_feat_path, '%06d_info.npy' % sampled_id)
        if not os.path.isfile(obj_info_file):
            break
        obj_info = np.load(obj_info_file, allow_pickle=True, encoding='latin1')
        instance_idlist_frames.append(obj_info.item()['instance_idlist'])
    all_instance_id = np.concatenate(instance_idlist_frames)
    max_instance_id, max_instance_num = np.max(all_instance_id), np.max(np.bincount(all_instance_id))
    # print(max_instance_id, max_instance_num,all_instance_id)
    # exit(0)
    aggregation_path = os.path.join(ANNO_DIR, scene_name, '%s.aggregation.json' % scene_name)
    aggre_file = json.load(open(aggregation_path, 'r'))
    if len(aggre_file['segGroups']) != max_instance_id:
        print(scene_name, max_instance_id, len(aggre_file['segGroups']))
    max_instance_id = len(aggre_file['segGroups'])

    obj_feat = np.zeros((max_instance_id + 1, max_instance_num, 2048))
    obj_coord = np.zeros((max_instance_id + 1, max_instance_num, 4))
    obj_size = np.zeros((max_instance_id + 1, max_instance_num))
    obj_depth = np.zeros((max_instance_id + 1, max_instance_num))
    camera_pose = np.zeros((max_instance_id + 1, max_instance_num, 16))
    instance_id = np.zeros((max_instance_id + 1, max_instance_num))
    frame_id = np.zeros((max_instance_id + 1, max_instance_num))
    instance_counter = [0 for ii in range(max_instance_id + 1)]

    for sampled_id in range(0, int(1e6), 100):
        cam_pose_file = os.path.join(cam_pose_path, '%06d.txt' % sampled_id)
        obj_feat_file = os.path.join(obj_feat_path, '%06d.npy' % sampled_id)
        obj_info_file = os.path.join(obj_feat_path, '%06d_info.npy' % sampled_id)
        depth_info_file = os.path.join(depth_feat_path, '%06d_info_depth.npy' % sampled_id)
        if not os.path.isfile(cam_pose_file):
            break
        came_pose = [x.strip().split(' ') for x in list(open(cam_pose_file))]
        came_pose = came_pose[0] + came_pose[1] + came_pose[2] + came_pose[3]
        came_pose = [float(x) for x in came_pose]
        feat = np.load(obj_feat_file)
        obj_info = np.load(obj_info_file, allow_pickle=True, encoding='latin1')
        depth_info = np.load(depth_info_file, allow_pickle=True, encoding='latin1')
        bbox, bbox_size, instance_idlist = obj_info.item()['bbox_list'], obj_info.item()['bbox_size'], obj_info.item()[
            'instance_idlist']
        for ii in range(len(instance_idlist)):
            ist_id, ist_n = instance_idlist[ii], instance_counter[instance_idlist[ii]]
            obj_feat[ist_id, ist_n, :] = feat[ii, :]
            obj_coord[ist_id, ist_n, :] = bbox[ii, :]
            obj_size[ist_id, ist_n] = bbox_size[ii]
            obj_depth[ist_id, ist_n] = depth_info.item()['depth'][ii]
            camera_pose[ist_id, ist_n, :] = came_pose
            instance_id[ist_id, ist_n] = instance_idlist[ii]
            frame_id[ist_id, ist_n] = sampled_id
            instance_counter[instance_idlist[ii]] += 1
    np.save(
        save_path, {'obj_feat': obj_feat, 'obj_coord': obj_coord, 'obj_size': obj_size, \
                    'camera_pose': camera_pose, 'instance_id': instance_id, 'obj_depth': obj_depth,
                    'frame_id': frame_id}
    )
