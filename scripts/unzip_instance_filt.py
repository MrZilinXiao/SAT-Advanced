from multiprocessing import Pool
import os
import argparse
from tqdm import tqdm


def mkdir_and_unzip(folder_path: str, zip_suffix='_2d-instance-filt.zip'):
    scene_name = os.path.basename(folder_path.rstrip('/'))
    zip_path = os.path.join(folder_path, scene_name + zip_suffix)
    assert os.path.isfile(zip_path)
    if os.path.isdir(os.path.join(folder_path, 'instance-filt')):
        print('Skipping {}...'.format(scene_name))
        return
    os.system('unzip -d {} {}'.format(folder_path, zip_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', type=str, default='/data/ScanNet/scans')
    cfg = parser.parse_args()

    scene_path_list = [os.path.join(cfg.scannet_path, folder) for folder in os.listdir(cfg.scannet_path)]
    with Pool(16) as p:
        r = list(tqdm(p.imap(mkdir_and_unzip, scene_path_list), total=len(scene_path_list)))
