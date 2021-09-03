import os
import numpy as np
# import tqdm
import argparse
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import json
from os import listdir
from os.path import isfile, join
import pickle
import glob
from scipy import ndimage

# install `vqa-maskrcnn-benchmark` from
# https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
"""
https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md
"""

"""
equivlant detectron2 settings here: 
https://github.com/clip-vil/CLIP-ViL/blob/master/CLIP-ViL-Direct/vqa/mcan_clip_grid_feature.py
https://github.com/clip-vil/CLIP-ViL/blob/c1d891776b58f40e4dc0ead6ccd1eab02c6ed45b/CLIP-ViL-Direct/vqa/pythia_clip_grid_feature.py#L73

please notice that maskecnn_benchmark does not support input_boxes training...

TODO: check the grid feature below is usable and trainable?
https://github.com/facebookresearch/grid-feats-vqa
"""


def load_detection_model(yaml_file, yaml_ckpt):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
    model.to("cuda")
    model.eval()
    return model


def _image_transform(image_path):
    img = Image.open(image_path)
    im = np.array(img).astype(np.float32)
    # handle a few corner cases
    if im.ndim == 2:  # gray => RGB
        im = np.tile(im[:, :, None], (1, 1, 3))
    if im.shape[2] > 3:  # RGBA => RGB
        im = im[:, :, :3]

    im = im[:, :, ::-1]  # RGB => BGR
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )  # H, W, C
    img = torch.from_numpy(im).permute(2, 0, 1)  # C, H, W, BGR
    return img, im_scale


def _process_feature_extraction(
        output, im_scales, feat_name='fc6'
):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []
    bbox_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep],
                cls_scores[keep],
                max_conf[keep]
            )

        keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        feat_list.append(feats[i][keep_boxes])
        bbox_list.append(output[0]["proposals"][i].bbox[keep_boxes])
    return feat_list, bbox_list


def extract_features(
        detection_model, image_path, input_boxes=None, feat_name='fc6'
):
    im, im_scale = _image_transform(image_path)
    if input_boxes is not None:
        if isinstance(input_boxes, np.ndarray):
            input_boxes = torch.from_numpy(input_boxes.copy())
        input_boxes *= im_scale
    img_tensor, im_scales = [im], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to('cuda')
    with torch.no_grad():
        output = detection_model(
            current_img_list, input_boxes=input_boxes)

    if input_boxes is None:  # will not be None
        feat_list, bbox_list = _process_feature_extraction(
            output, im_scales, feat_name)
        feat = feat_list[0].cpu().numpy()
        bbox = bbox_list[0].cpu().numpy() / im_scale
    else:
        feat = output[0][feat_name].cpu().numpy()
        bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale

    return feat, bbox


def main():
    parser = argparse.ArgumentParser()
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
    # parser.add_argument(
    #     "--imdb_file", type=str,
    #     default='/private/home/ronghanghu/workspace/pythia/data/imdb/m4c_textvqa/imdb_train_ocr_en.npy',
    #     help="The imdb to extract features"
    # )
    parser.add_argument(
        "--image_dir", type=str,
        default='/data/ScanNet/tasks/scannet_frames_25k/scannet_frames_25k/',
        help="The directory containing images"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default='',
        help="The directory to save extracted features"
    )
    parser.add_argument(
        "--bbox_dir", type=str,
        default='',
        help="The directory containing images"
    )
    parser.add_argument('--gpu', default='0', help='gpu id')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    DETECTION_YAML = args.detection_cfg
    DETECTION_CKPT = args.detection_model
    # IMDB_FILE = args.imdb_file
    IMAGE_DIR = args.image_dir
    GTMAP_DIR = args.bbox_dir
    SAVE_DIR = args.save_dir

    detection_model = load_detection_model(DETECTION_YAML, DETECTION_CKPT)
    print('Faster R-CNN OCR features')
    print('\tsaving to', SAVE_DIR)

    scene_list = glob.glob('%s/*' % GTMAP_DIR)
    for scene_file in tqdm(scene_list):
        scene_name = scene_file.split('/')[-1]
        for sampled_id in range(0, int(1e6), 100):
            # for sampled_id in range(0,int(1e6),10):
            bbox_list, instance_idlist = [], []
            image_path = os.path.join(IMAGE_DIR, scene_name, 'color/%d.jpg' % sampled_id)
            # image_path = os.path.join(IMAGE_DIR,scene_name,'color/%06d.jpg'%sampled_id)   ## 570, 93-2,173-1,251,705-1,412-1 has no 0th frame
            if not os.path.isfile(image_path):
                break
            instance_file = os.path.join(scene_file, 'instance-filt/%d.png' % sampled_id)
            os.system('mkdir -p %s' % (os.path.join(SAVE_DIR, scene_name)))
            save_info_path = os.path.join(SAVE_DIR, scene_name, '%06d_info.npy' % sampled_id)
            save_feat_path = os.path.join(SAVE_DIR, scene_name, '%06d.npy' % sampled_id)
            if os.path.isfile(save_feat_path):
                break
            instance = cv2.imread(instance_file)
            assert ((instance[:, :, 0] == instance[:, :, 1]).all() and (instance[:, :, 1] == instance[:, :, 2]).all())
            instance = instance[:, :, 0]
            for instance_id in np.unique(instance):
                # 不是用投影，而是用instance文件的label标注，比较妙

                s = ndimage.generate_binary_structure(2, 2)  ## 8-way connect, instead of the default 4-way
                labeled_array, numpatches = ndimage.label(np.array(instance == instance_id, dtype=int),
                                                          s)  ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
                largest_cc_mask_xy = np.where(labeled_array == (np.bincount(labeled_array.flat)[1:].argmax() + 1))
                x_min, x_max = largest_cc_mask_xy[1].min(), largest_cc_mask_xy[1].max()
                y_min, y_max = largest_cc_mask_xy[0].min(), largest_cc_mask_xy[0].max()
                bbox_list.append([x_min, y_min, x_max, y_max])
                instance_idlist.append(instance_id)

            bbox_list = np.array(bbox_list, dtype=np.float32)
            instance_idlist = np.array(instance_idlist, dtype=int)

            if len(bbox_list) > 0:
                extracted_feat, _ = extract_features(
                    detection_model, image_path, input_boxes=bbox_list
                )
            else:
                extracted_feat = np.zeros((0, 2048), np.float32)

            bbox_size = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])
            np.save(
                save_info_path, {'bbox_list': bbox_list, 'instance_idlist': instance_idlist, 'bbox_size': bbox_size}
            )
            np.save(save_feat_path, extracted_feat)


if __name__ == '__main__':

    main()
