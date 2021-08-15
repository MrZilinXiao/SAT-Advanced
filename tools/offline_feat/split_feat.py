"""
why this?
because numpy mmap exceptions: "Array can't be memory-mapped: Python objects in dtype."
so we have to split dicts in npy file into multiple npy files.
also we may specify the keys of feat to reduce cache memory consumption!
"""
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

SOURCE_PATH = '/data/meta-ScanNet/CLIP-npy/scannet_frames_25k_CLIP_RN5016'
DEST_PATH = '/data/meta-ScanNet/split_feat'
KEYS = ['clip_region_feat', 'clip_scaled_region_feat', 'obj_feat',
        'obj_coord', 'obj_size', 'camera_pose', 'instance_id',
        'obj_depth', 'frame_id']
SCENE_LEN = 1513


def get_workers(x=-1):
    if x == -1:
        x = mp.cpu_count()
    return x


def get_npy_list(npy_dir):
    res = [os.path.join(npy_dir, p) for p in os.listdir(npy_dir) if p.endswith('.npy')]
    assert len(res) == SCENE_LEN, "Incomplete Offline Features!"
    return res


def split_npy(npy_path):
    res = np.load(npy_path, allow_pickle=True)  # Python dicts inside
    scene_name = os.path.basename(npy_path)[:-4]  # scene0685_01
    for k in KEYS:
        res_dump = res.item()[k]
        dest_path = os.path.join(DEST_PATH, scene_name + '_' + k + '.npy')  # DEST_PATH/scene0685_01_clip_region_feat.npy
        np.save(dest_path, res_dump, allow_pickle=False)


if __name__ == '__main__':
    scenes_list = get_npy_list(SOURCE_PATH)
    with mp.Pool(get_workers()) as p:
        _ = list(tqdm(p.imap(split_npy, scenes_list), total=len(scenes_list)))
