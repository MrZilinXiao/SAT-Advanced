"""
utils for online FasterRCNN ROI feature extraction
"""

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import torch
import numpy as np


def load_detection_model(yaml_file, yaml_ckpt, device='cpu'):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
    model.to(device)
    model.train()
    return model


class FasterRCNNExtractor:
    """
    TODO: This RCNN implementation wouln't allow training for bbox!
    """
    def __init__(self, yaml_file, yaml_ckpt, device='cpu'):
        self.model = load_detection_model(yaml_file, yaml_ckpt, device)

    # def extract_features(
    #         self, image_path, input_boxes=None, feat_name='fc6'
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

    def get_roi_feat(self, img_list, im_scale_list, obj_bbox_list, max_context_size=51):
        """
        img_list: list of torch Tensor (C, H, W)
        obj_bbox_list: list of a list containing <obj, bbox>

        return: torch Tensor (max_context_size, feat_size)
        """
        max_obj_count = -1
        frame_bbox_list = []  # [(frame 1)[bbox1, bbox2, ...] ]
        for frame_objs in obj_bbox_list:
            bbox_list = []
            for obj, bbox in frame_objs:
                max_obj_count = max(max_obj_count, obj)
                bbox_list.append(np.array(bbox, dtype=np.float32))
            frame_bbox_list.append(bbox_list)


