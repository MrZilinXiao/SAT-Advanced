from multiprocessing import Pool
from tqdm import tqdm
import os

SCANS_PATH = '/data/ScanNet/scans'
OUTPUT_PATH = '/data/ScanNet/uncompressed'
BIN_PATH = '/data/public_code/ScanNet/SensReader/c++/sens'


def uncompress(scene):
    target_path = os.path.join(OUTPUT_PATH, scene)
    sens_path = os.path.join(SCANS_PATH, scene, "{}.sens".format(scene))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    os.system("{} {} {} >/dev/null 2>&1".format(BIN_PATH, sens_path, target_path))


if __name__ == '__main__':
    # list all scenes
    scene_list = list(sorted(os.listdir(SCANS_PATH)))
    # scene_sens_path_list = [os.path.join(SCANS_PATH, s, "{}.sens".format(s)) for s in scene_list]
    with Pool(12) as p:
        r = list(tqdm(p.imap(uncompress, scene_list), total=len(scene_list)))
