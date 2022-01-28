from abc import ABC

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

# from maskrcnn_benchmark.modeling

import numpy as np
import torch
from PIL import Image
import cv2
from models.offline_extractor import BaseExtractor


def load_detection_model(yaml_file, yaml_ckpt):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
    model.to("cuda")
    model.eval()
    return model


def _image_transform(im):
    # img = Image.open(image_path)
    # im = np.array(img).astype(np.float32)
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


# def extract_features(
#         detection_model, image_path, input_boxes=None, feat_name='fc6'
# ):
#     im, im_scale = _image_transform(image_path)
#     if input_boxes is not None:
#         if isinstance(input_boxes, np.ndarray):
#             input_boxes = torch.from_numpy(input_boxes.copy())
#         input_boxes *= im_scale
#     img_tensor, im_scales = [im], [im_scale]
#     current_img_list = to_image_list(img_tensor, size_divisible=32)
#     current_img_list = current_img_list.to('cuda')
#     with torch.no_grad():
#         output = detection_model(
#             current_img_list, input_boxes=input_boxes)
#
#     if input_boxes is None:  # will not be None
#         feat_list, bbox_list = _process_feature_extraction(
#             output, im_scales, feat_name)
#         feat = feat_list[0].cpu().numpy()
#         bbox = bbox_list[0].cpu().numpy() / im_scale
#     else:
#         feat = output[0][feat_name].cpu().numpy()
#         bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale
#
#     return feat, bbox


class RCNNExtractor(BaseExtractor, ABC):
    def __init__(self, args):
        super().__init__()
        self.model = load_detection_model(args.detection_cfg, args.detection_model)
        self.preprocessor = _image_transform  # feed in img_data
        self.model.eval()

    def get_bbox_feature(self, img, bbox_list, feat_name='fc6'):
        """
        img should be loaded via:
            img = Image.open(image_path)
        """
        img = np.array(img).astype(np.float32)

        im, im_scale = self.preprocessor(img)
        if bbox_list is not None:
            if isinstance(bbox_list, np.ndarray):
                bbox_list = torch.from_numpy(bbox_list.copy())
            bbox_list *= im_scale

        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.model(
                current_img_list, input_boxes=bbox_list)  # this might not align with original maskrcnn-benchmark...

        if bbox_list is None:  # will not be None
            feat_list, bbox_list = _process_feature_extraction(
                output, im_scales, feat_name)
            feat = feat_list[0].cpu().numpy()
            # bbox = bbox_list[0].cpu().numpy() / im_scale
        else:
            feat = output[0][feat_name].cpu().numpy()
            # bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale

        # return feat, bbox
        return feat
    # feat: np.array [bbox_size, feat_size (2048)], bbox: useless


if __name__ == '__main__':
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    args = {
        'detection_cfg': '/data/pretrained_models/detectron_model.yaml',
        'detection_model': '/data/pretrained_models/detectron_model.pth'
    }
    args = argparse.Namespace(**args)
    extractor = RCNNExtractor(args)

    dummy_img = Image.open('/data/ScanNet/uncompressed/scene0000_00/frame-000000.color.jpg')
    dummy_bbox = np.array([[30, 30, 300, 300]], dtype=np.float32)

    print(extractor.get_bbox_feature(dummy_img, dummy_bbox))
