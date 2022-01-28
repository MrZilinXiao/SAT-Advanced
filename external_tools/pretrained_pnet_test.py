import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from loguru import logger


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):  # 这个是resume用的
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():  # load model only
        # if key != 'model':
        #     continue
        kwargs = {}
        if key == 'model_state_dict':
            kwargs['strict'] = False
        load_status = value.load_state_dict(checkpoint[key], **kwargs)
        if load_status is not None and str(load_status) != '<All keys matched successfully>':
            logger.warning("Caught some errors when loading state_dict for {}:\n".format(key) +
                           f"missing keys: {load_status.missing_keys}\nunexpected_keys: {load_status.unexpected_keys}")


class SAT_PNet2(nn.Module):
    def __init__(self, obj_latent_size, normal_channel=True):
        super(SAT_PNet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # these modules remain untouched since we need their pretrained weights
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        # keep only layer-1
        self.adapter = nn.Linear(512, obj_latent_size)

    def forward(self, xyz):  # xyz & feat [BS, C, num_obj]
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        # return x, l3_points
        return x

# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 3 if normal_channel else 0
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
#                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
#         self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
#                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
#         self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, num_class)
#
#     def forward(self, xyz):  # xyz & feat
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)
#
#         return x, l3_points


if __name__ == '__main__':
    model = SAT_PNet2(768)  # [batch_size (or num_points), 6, num_points]
    # ckpt = torch.load('/data/public_code/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth')
    load_state_dicts(
        checkpoint_file='/data/public_code/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth',
        model_state_dict=model)

    model = model.cuda()
    # original SAT batch input
    bs, obj_cnt, num_point, xyz_feat = 2, 2, 1024, 6  # too large for this PNet
    batch_objects = torch.randn((bs, obj_cnt, num_point, xyz_feat), dtype=torch.float)  # B, N_objects, N_Points, 3 + C

    # make some adaption to new pnet2
    batch_objects = batch_objects.view(bs * obj_cnt, num_point, xyz_feat).permute(0, 2, 1).cuda()
    # test forward it
    output = model(batch_objects)
    print(output.shape)


    # TODO 检查通道是否与原setting对齐
