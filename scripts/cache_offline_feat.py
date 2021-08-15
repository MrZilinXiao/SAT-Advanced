"""
Each time after restarting the OS:
sudo mount -o size=32000M  -o  remount  /dev/shm
"""
import os
import sys

sys.path.append(os.getcwd())

import argparse
from utils import share_array
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time

SCENE_LEN = 1513
CACHE_PATH = None
CACHE_FILE_PREFIX = 'sharearray_'
KEYS = None


def _check(res_list):
    res_set = set()
    for k in KEYS:
        res_set.add(len([p for p in res_list if p.endswith(k + '.npy')]))
    return len(res_set) == 1 and list(res_set)[0] == SCENE_LEN


def get_workers(x):
    if x == -1:
        x = mp.cpu_count()
    return x


def get_npy_list(npy_path):
    res = [os.path.join(npy_path, p) for p in os.listdir(npy_path) if p.endswith('.npy')]
    _check(res)  # some simple check
    return res


def cache_array(scene_path):
    scene_name = os.path.basename(scene_path)[:-4]  # scene0685_01_clip_region_feat
    share_array.cache(scene_name, lambda: np.load(scene_path, allow_pickle=False),
                      shm_path=CACHE_PATH,
                      prefix=CACHE_FILE_PREFIX)


def clean_up_cache():
    del_list = [p for p in os.listdir(CACHE_PATH) if os.path.basename(p).startswith(CACHE_FILE_PREFIX)]
    for file in del_list:
        try:
            os.remove(file)
        except FileNotFoundError:
            print("Not Found: {}".format(file))
    print('Clean up Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline-2d-feat', type=str,
                        default='/data/meta-ScanNet/split_feat/',
                        help='The disk path for offline 2D features')
    parser.add_argument('--cache-path', type=str, default='/dev/shm',
                        help='Cache target path, best be `/dev/shm` or other high-speed mount point (SSD, etc.)')
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--keys', nargs='+', default=['clip_region_feat', 'clip_scaled_region_feat', 'obj_feat',
                                                      'obj_coord', 'obj_size', 'camera_pose', 'instance_id',
                                                      'obj_depth', 'frame_id'])
    args = parser.parse_args()
    num_workers = get_workers(args.workers)
    CACHE_PATH = args.cache_path
    KEYS = args.keys
    try:
        scenes_list = get_npy_list(args.offline_2d_feat)
        if not args.debug:
            with mp.Pool(num_workers) as p:
                _ = list(tqdm(p.imap(cache_array, scenes_list), total=len(scenes_list)))
                # automatic join & close
        else:
            for scene_path in tqdm(scenes_list):
                cache_array(scene_path)

        print("Done Caching! Happy Training! Ctrl+C to clean up cache.")

        while True:  # ugly infinite loop
            try:
                time.sleep(1e9)
            except KeyboardInterrupt:
                break
        print('Done!')
    finally:
        print('Cleaning up cache...')
        clean_up_cache()
