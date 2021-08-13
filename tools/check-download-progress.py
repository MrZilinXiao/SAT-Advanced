import os


def check_file(dic):
    url, out_file = dic['url'], dic['path']
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return os.path.isfile(out_file)


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
    complete_cnt = 0
    for file in download_list:
        complete_cnt = complete_cnt + 1 if check_file(file) else complete_cnt

    print(complete_cnt / len(download_list))