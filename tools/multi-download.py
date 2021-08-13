"""
A modified ScanNet Downloader supporting multi-process pool
"""
import os
# import urllib
import urllib.request
import tempfile
from multiprocessing import Pool
from tqdm import tqdm
import socket
import time
socket.setdefaulttimeout(30)


def download_file(dic, retry_times=5):
    url, out_file = dic['url'], dic['path']
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        while retry_times > 0:
            try:
                print('\t' + url + ' > ' + out_file)
                fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
                f = os.fdopen(fh, 'w')
                f.close()
                # urllib.urlretrieve(url, out_file_tmp)
                urllib.request.urlretrieve(url, out_file_tmp)
                os.rename(out_file_tmp, out_file)
                break
            except Exception:
                print('retrying {}...'.format(url))
                retry_times -= 1
                time.sleep(1)

    else:
        print('WARNING: skipping download of existing file ' + out_file)


def read_filelist(file):
    ret = list()
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            assert len(line) == 2
            ret.append({'url': line[0].strip(), 'path': line[1].strip()})
    return ret


if __name__ == '__main__':
    download_list = read_filelist('scannet_filelist.txt')
    with Pool(15) as p:
        r = list(tqdm(p.imap(download_file, download_list), total=len(download_list)))
    # pool = Pool(10)
    # pool.map(download_file, download_list)
    # pool.close()
    # pool.join()
