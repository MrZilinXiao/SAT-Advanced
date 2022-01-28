from glob import glob
import json
from multiprocessing import Pool
from tqdm import tqdm


def bbox_filter(bbox, delta_px=10):
    """
    |x1 - x2| or |y1 - y2| should be greater than `delta_px`
    """
    return abs(bbox[0] - bbox[2]) > delta_px and abs(bbox[1] - bbox[3]) > delta_px


def submit_job(scene_kf_json):
    with open(scene_kf_json, 'r') as f:
        scene_kf = json.load(f)
    for i in range(len(scene_kf['key_frames'])):
        kf = scene_kf['key_frames'][i]

        instance_ids, bboxes = [], []
        # filter out invalid box
        for ins_id, bbox in zip(kf['id'], kf['bbox']):
            if bbox_filter(bbox, 10):
                instance_ids.append(ins_id)
                bboxes.append(bbox)

        kf['id'] = instance_ids
        kf['bbox'] = bboxes

    with open(scene_kf_json, 'w') as f:
        json.dump(scene_kf, f, indent=2)

    return None


if __name__ == '__main__':
    # 首先做所有frames的key frame selection
    # dataset = CamScannetScene('/data/ScanNet/uncompressed/', '/data/ScanNet/scans/', 'scene0000_00')
    # key_frame_ids_for_obj_4 = dataset.compute_key_frame(dataset.instance2frame_range_dict[4])
    # print("instance 4 exists in frame list {}, where key frames are {}.".format(dataset.instance2frame_range_dict[4],
    #                                                                             key_frame_ids_for_obj_4))
    # 尝试对每个instance做key frame selection
    scene_list = list(sorted(glob('/data/meta-ScanNet/key_frame_bbox/*.kf.bbox.json')))
    with Pool(12) as p:
        r = list(tqdm(p.imap(submit_job, scene_list), total=len(scene_list)))
