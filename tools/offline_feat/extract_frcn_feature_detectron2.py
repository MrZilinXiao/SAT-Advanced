from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
import argparse
import torch


def load_detection_model(yaml_file, yaml_ckpt):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
    model.to("cuda")
    model.eval()
    return model


if __name__ == '__main__':
    model = load_detection_model('/data/pretrained_models/detectron_model.yaml',
                                 '/data/pretrained_models/detectron_model.pth')
    print(model)
