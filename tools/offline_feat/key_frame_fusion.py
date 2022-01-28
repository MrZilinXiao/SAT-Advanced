import torch
import argparse
import os
import glob
import numpy as np
from torch.nn.functional import grid_sample
from utils.three_d_fusion import generate_grid

REFERENCE = '/data/ScanNet/scans/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_input_path', type=str, default='/data/ScanNet/uncompressed/')
    parser.add_argument('--feat_input_path', type=str, default='/data/meta-ScanNet/key_frame_feature/clip/')
    parser.add_argument('--output_path', type=str, default='/data/meta-ScanNet/aggre_feature/')
    parser.add_argument('--scene_name', type=str, default=None)

    return parser.parse_args()


# is this computation heavy? need further profiling...
def back_project(coords, origin, voxel_size, feats, KRcam):
    """
    Unproject the image features to form a 3D (sparse) feature volume
    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix  -> pose matrix is projection matrix, P = K[R|t]
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    """
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]  # N_views x C x H x W
        proj_batch = KRcam[:, batch]  # N_views x 4 x 4

        grid_batch = coords_batch * voxel_size + origin_batch.float()  # origin is a shift on coords
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

        # Project grid
        im_p = proj_batch @ rs_grid  # direct multiply
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()

        # aggregate multi view  -> 3D Feature Volume
        features = features.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count


def aggre_obj_feature(args, scene_name: str):
    """
    aggregate an instance object's feature using unprojection
    步骤：
    1. 读当前scene所有keyframe，按object_id group-by
    2. unproject -> 3D feature volume
    3. 3D sparse conv (only test, 这一步放到MMT fusion一起train)
    """
    keyframes_npy = glob.glob(os.path.join(args.feat_input_path, f'{scene_name}_kf_*.npy'))
    kf_data_dict = dict()
    for kf_path in keyframes_npy:
        kf = np.load(kf_path, allow_pickle=True)
        kf_id = kf.item()['key_frame_id']

        kf_data_dict[kf_id]: dict = kf.item()  # strip the ndarray `.item()`

        # add pose info
        pose_path = os.path.join(args.pose_input_path, scene_name, 'frame-{frame_id:0>6d}.pose.txt'.format(frame_id=kf_id))
        kf_data_dict[kf_id]['pose'] = np.loadtxt(pose_path).astype(np.float32)  # 4x4 pose

    # get unique instance list
    unique_instance_id = set()
    for v in kf_data_dict:
        unique_instance_id.update(v['instance_id'])

    # for each unique instance, collect their key frame's feature & pose
    for instance_id in unique_instance_id:
        feat_list, pose_list = [], []
        for kf, kf_data in kf_data_dict.items():
            # if instance_id not in kf_data['instance_id']:
            #     continue
            try:
                target_idx = kf_data['instance_id'].index(instance_id)
            except ValueError:  # remove duplicated index
                continue
            feat_list.append(kf_data['feat'][target_idx])
            pose_list.append(kf_data['pose'][target_idx])

        coords = generate_grid([96, 96, 96], 1)[0]  #


    pass


if __name__ == '__main__':
    args = parse_args()
    if args.scene_name is None:
        scene_list = os.listdir(REFERENCE)
    else:
        scene_list = [args.scene_name]  # debug mode
