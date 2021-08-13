"""
之前的脚本没有.rstrip()，后处理一下；
"""
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    cfg = parser.parse_args()
    for root, _, files in os.walk(cfg.dir):
        for file in files:
            file: str
            file_path = os.path.join(root, file)
            if file.startswith('tmp'):
                os.remove(file_path)
                print('Removed ', file_path)
            elif file.endswith('\n'):
                new_file_path = file_path.rstrip()
                os.rename(file_path, new_file_path)
                print('Renamed {} to {}'.format(file_path, new_file_path))
