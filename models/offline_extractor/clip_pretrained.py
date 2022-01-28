from abc import ABC

import torch
import clip
# from tqdm import tqdm
# import numpy as np
from PIL import Image
# import torchvision.transforms as transforms
from models.offline_extractor import BaseExtractor
from torchvision.transforms import *
BICUBIC = InterpolationMode.BICUBIC

# loader = transforms.Compose([transforms.ToTensor()])


class CLIPExtractor(BaseExtractor, ABC):
    def __init__(self, args, device='cuda'):
        super().__init__()
        self.model, self.preprocessor = clip.load(args.clip_type, device=device)
        self.device = device
        self.model.eval()

        # self.preprocessor = torch.nn.Sequential(
        #     Resize(384, interpolation=BICUBIC),
        #     CenterCrop(384),
        #     # lambda image: image.convert("RGB"),
        #     # ToTensor(),
        #     ConvertImageDtype(torch.float),
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # )

    # @torch.no_grad()
    # def get_bbox_feature_numpy(self, img, bbox_list):
    #     # img loaded via cv2.imread
    #     regions = []
    #     for bbox in bbox_list:
    #         x1, y1, x2, y2 = bbox.astype(int)
    #         # region = img[y1: y2, x1: x2]  # cv2: H, W, C
    #         # region = torch.from_numpy(img[y1: y2, x1: x2]).to('cuda')
    #         regions.append(self.preprocessor(img.crop((x1, y1, x2, y2))))
    #
    #     regions = torch.stack(regions, dim=0).to(self.device)  # bbox_num, C, H, W
    #
    #     regions_feat = self.model.encode_image(regions)
    #
    #     return regions_feat.cpu().numpy()

    @torch.no_grad()
    def get_bbox_feature(self, img, bbox_list):
        """
        img should be loaded via:
            img = Image.open(image_path)
        """
        # return self.get_bbox_feature_numpy(img, bbox_list)

        regions = []  # list of C, H, W
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox.astype(int)
            regions.append(self.preprocessor(img.crop((x1, y1, x2, y2))))  # grid-based CLIP feature

        regions = torch.stack(regions, dim=0).to(self.device)  # bbox_num, C, H, W
        # with torch.no_grad():
        regions_feat = self.model.encode_image(regions)

        return regions_feat.cpu().numpy()
    # regions_feat: [N, clip_dim]


if __name__ == '__main__':
    import argparse
    import os
    from PIL import Image
    import numpy as np
    import cv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = {
        'clip_type': 'RN50x16',
    }
    args = argparse.Namespace(**args)
    extractor = CLIPExtractor(args)

    dummy_img = cv2.imread('/data/ScanNet/uncompressed/scene0000_00/frame-000000.color.jpg')
    dummy_bbox = np.array([[30, 30, 300, 300], [2, 30, 300, 300]], dtype=np.float32)

    print(extractor.get_bbox_feature(dummy_img, dummy_bbox))

