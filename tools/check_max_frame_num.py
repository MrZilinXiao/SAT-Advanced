"""
This tool is used to check the max num of 2D frames related to a target object
"""
import os
from functools import partial

import numpy as np
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset

from tqdm import tqdm
from loguru import logger
import torch
import random

from in_out.frcn_read import frcn_image_transform
from in_out.pt_datasets.utils import pad_samples, sample_scan_object, instance_labels_of_context, \
    check_segmented_object_order
from utils import share_array
from collections import defaultdict


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, offline_2d_feat, max_seq_len, points_per_object, max_distractors, args,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, pretrain=False, context_object=False, feat2dtype=None, addlabel_words=False,
                 num_class_dim=525, evalmode=False, split='train'):
        self.args = args
        self.references = references
        self.scans = scans  # should be shared across experiments !!
        self.vocab = vocab
        self.offline_2d_feat = offline_2d_feat
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation

        # SAT added parts below
        self.pretrain = pretrain
        # self.context_object = context_object  # bool: 是否加入context_objects
        self.feat2dtype = feat2dtype
        # self.max_2d_view = 5
        # self.addlabel_words = addlabel_words  # remove duplicated flags
        self.num_class_dim = num_class_dim
        self.evalmode = evalmode

        if not args.use_clip_language:  # TextBERT
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
            assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]
        else:  # clip-BERT
            import clip
            self.clip_tokenizer = partial(clip.tokenize, truncate=True)

        if not check_segmented_object_order(scans):
            raise ValueError

        self.extract_text = args.extract_text
        self.use_offline_language = len(args.offline_language_path) > 0
        self.split = split
        self.use_online_visual = len(args.rgb_path) > 0
        self.max_frames = 30

        if self.use_offline_language:
            # assert args.language_type is not None
            if args.offline_language_type == 'clip':
                self.language_suffix = 'clip_text_feat'
            elif args.offline_language_type == 'bert':
                self.language_suffix = 'bert_text_feat'
            else:
                raise NotImplemented('Unknown language type...')
            logger.info("Using offline language feature from {}...".format(args.offline_language_path))

    def __len__(self):
        return len(self.references)
        # return int(len(self.references)//3.293)

    def get_reference_data(self, index):
        # ScannetScan Object, ThreeDObject Object, [24 + 2] ndarray (from self.vocab), list of word str, bool flag of is_nr3d
        ref = self.references.loc[index]  # pandas dataframe, save the index would work?
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, ref['tokens'], is_nr3d

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    @staticmethod
    def _cache_reader(scene_name, key_name, cache_path='/dev/shm'):
        _id = scene_name + '_' + key_name
        return share_array.read_cache(_id, shm_path=cache_path)

    def _load_split_offline_npy(self, scene_name, key_name):
        _path = os.path.join(self.args.offline_2d_feat, scene_name + '_' + key_name + '.npy')
        return np.load(_path)

    # @profile
    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, text_tokens, is_nr3d = self.get_reference_data(index)
        # ScannetScan Object, ThreeDObject Object, [24 + 2] ndarray (from self.vocab), list of word str, bool flag of is_nr3d

        # TextBERT tokenize
        if not self.args.use_clip_language:
            token_inds = torch.zeros(self.max_seq_len, dtype=torch.long)
            indices = self.bert_tokenizer.encode(
                ' '.join(text_tokens), add_special_tokens=True)
            indices = indices[:self.max_seq_len]  # TODO: encode之后取前max_seq_len个，这会丢失EOS嘛！
            token_inds[:len(indices)] = torch.tensor(indices)
            token_num = torch.tensor(len(indices), dtype=torch.long)
        else:
            clip_indices = self.clip_tokenizer(' '.join(text_tokens))
            clip_indices = clip_indices.squeeze(0)  # have to remove batch_dim, shape: [77] on cpu
            token_num = torch.sum(clip_indices != 0, dtype=torch.long)

        res['csv_index'] = index  # this index varies in train / test split

        if self.use_offline_language:
            res['txt_emb'] = np.load(os.path.join(self.args.offline_language_path, '{}_{}_{}.npy'.format(index, self.language_suffix, self.split)))

        if self.extract_text:  # for temporary extractor
            if not self.args.use_clip_language:
                res['tokens'] = tokens
                res['token_inds'] = token_inds.numpy().astype(np.int64)  # model takes these
                res['token_num'] = token_num.numpy().astype(np.int64)
            else:
                res['clip_inds'] = clip_indices
            # TODO: please make sure that we wouldn't change the data reading strategy in the future!!
            return res
        # if self.pretrain:
        #     ## entire seq replace for now
        #     contra_rand = random.random()
        #     if False:
        #         tag_pollute = torch.tensor([contra_rand < 0.25]).long()
        #         query_pollute = torch.tensor([contra_rand > 0.75]).long()
        #         contra_pollute = (tag_pollute + query_pollute).clamp(max=1)
        #     else:
        #         tag_pollute, query_pollute, contra_pollute = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(
        #             1).long()
        #     res['tag_pollute'] = tag_pollute.numpy().astype(np.int64)
        #     res['query_pollute'] = query_pollute.numpy().astype(np.int64)
        #     res['contra_pollute'] = contra_pollute.numpy().astype(np.int64)
        #     ##
        #     token_inds, mlm_label = self.random_word(token_inds, self.bert_tokenizer.vocab, mask_prob=0.15

        # Make a context of distractors
        context = self.prepare_distractors(scan, target)  # default max: 51

        # Add target object in 'context' list at random position
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])  ## (N, 1024, 6)

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        if self.object_transformation is not None:
            # samples = self.object_transformation(samples)
            samples, offset = self.object_transformation(samples)
            res['obj_offset'] = np.zeros((self.max_context_size, offset.shape[1])).astype(np.float32)
            res['obj_offset'][:len(offset), :] = offset.astype(np.float32)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # TODO: read in image patches if use_clip_visual is enabled, 用于计划2
        objfeat_2d = False if self.use_online_visual else True   # set objfeat_2d to False if planning to use online RoI
        # TODO: 检查对obj_feat2d的引用情况

        if not self.args.offline_cache:  # no caching...
            # context_2d = np.load('%s/%s.npy' % (self.offline_2d_feat, scan.scan_id),
            #                      allow_pickle=True, encoding='latin1')
            # choose objfeat_2d based on args.feat2d
            if self.args.feat2d.startswith('ROI'):
                objfeat_2d = self._load_split_offline_npy(scan.scan_id, 'obj_feat') if objfeat_2d else False
                featdim = 2048
            elif self.args.feat2d.startswith('CLIP_add'):
                objfeat_2d = self._load_split_offline_npy(scan.scan_id, 'clip_region_feat') + \
                             self._load_split_offline_npy(scan.scan_id, 'clip_scaled_region_feat') if objfeat_2d else False
                featdim = 768
            elif self.args.feat2d.startswith('CLIP'):  # feat that do not norm!! should consider?
                objfeat_2d = self._load_split_offline_npy(scan.scan_id, 'clip_region_feat') if objfeat_2d else False # also do not norm...
                featdim = 768
            else:
                raise NotImplemented("Not recognized feat2d keys: {}".format(self.args.feat2d))

            # bbox_2d = context_2d.item()['obj_coord']
            bbox_2d = self._load_split_offline_npy(scan.scan_id, 'obj_coord')

            # bboxsize_2d = context_2d.item()['obj_size']
            # obj_depth = context_2d.item()['obj_depth']
            # campose_2d = context_2d.item()['camera_pose']
            campose_2d = self._load_split_offline_npy(scan.scan_id, 'camera_pose')
            # ins_id_2d = context_2d.item()['instance_id']
            ins_id_2d = self._load_split_offline_npy(scan.scan_id, 'instance_id')
            frame_id_2d = self._load_split_offline_npy(scan.scan_id, 'frame_id')

        else:  # read cache
            if self.args.feat2d.startswith('ROI'):
                objfeat_2d = self._cache_reader(scan.scan_id, 'obj_feat') if objfeat_2d else False  # also do not norm...
                featdim = 2048
            elif self.args.feat2d.startswith('CLIP_add'):
                objfeat_2d = self._cache_reader(scan.scan_id, 'clip_region_feat') + self._cache_reader(scan.scan_id,
                                                                                                       'clip_scaled_region_feat') if objfeat_2d else False
                featdim = 768
            elif self.args.feat2d.startswith('CLIP'):  # feat that do not norm!! should consider?
                objfeat_2d = self._cache_reader(scan.scan_id, 'clip_region_feat') if objfeat_2d else False
                featdim = 768
            else:
                raise NotImplemented("Not recognized feat2d keys: {}".format(self.args.feat2d))

            bbox_2d = self._cache_reader(scan.scan_id, 'obj_coord')
            # bboxsize_2d = self._cache_reader(scan.scan_id, 'obj_size')
            # obj_depth = self._cache_reader(scan.scan_id, 'obj_depth')
            campose_2d = self._cache_reader(scan.scan_id, 'camera_pose')
            ins_id_2d = self._cache_reader(scan.scan_id, 'instance_id')
            frame_id_2d = self._cache_reader(scan.scan_id, 'frame_id')

        if self.args.clsvec2d:
            featdim += self.num_class_dim

        if self.use_online_visual:  # if not loading offline features, we provide loaded images and bbox for pretrained Faster-RCNN
            feat_2d = None
        else:
            feat_2d = np.zeros((self.max_context_size, featdim)).astype(np.float32)

        coords_2d = np.zeros((self.max_context_size, 4 + 12)).astype(np.float32)
        # coords_2d = np.zeros((self.max_context_size, 4+1+12)).astype(np.float32)

        selected_2d_idx = 0
        # selected_2d_idx = [random.randint(0, max(0,int((ins_id_2d[ii,:]!=0).astype(np.float32).sum())-1)) for ii in range(ins_id_2d.shape[0])]
        ##
        selected_context_id = [o.object_id + 1 for o in context]  # background included, so +1
        # print(scan.scan_id,objfeat_2d.shape,selected_context_id)

        # first choose, all view-0 on dim 1
        if self.use_online_visual:
            selected_objfeat_2d = None
        else:
            selected_objfeat_2d = objfeat_2d[selected_context_id, selected_2d_idx, :]  ## ROI feat_2d

        selected_bbox_2d = bbox_2d[selected_context_id, selected_2d_idx, :]
        # selected_bboxsize_2d = bboxsize_2d[selected_context_id, selected_2d_idx]
        # selected_obj_depth = obj_depth[selected_context_id, selected_2d_idx]
        selected_campose_2d = campose_2d[selected_context_id, selected_2d_idx, :]
        # selected_ins_id_2d = ins_id_2d[selected_context_id, selected_2d_idx]
        selected_frame_id_2d = frame_id_2d[selected_context_id, selected_2d_idx]

        # if True:  # use random selected_2d_idx, instead of 0 (dummy if True, removed)
        # context_to_view = dict()  # 记录对每个context选了第几个view (随机的)

        for ii in range(len(selected_context_id)):
            cxt_id = selected_context_id[ii]
            view_id = random.randint(0, max(0, int((ins_id_2d[cxt_id, :] != 0).astype(
                np.float32).sum()) - 1))  # 对每个context object 随机一个有效的view，view_id保证index的frame都是有效的
            # context_to_view[cxt_id] = view_id
            if self.use_online_visual:
                selected_frame_id_2d[ii] = frame_id_2d[cxt_id, view_id]  # need frame id to reference RGB image
            else:
                selected_objfeat_2d[ii, :] = objfeat_2d[cxt_id, view_id, :]  ## ROI feat_2d
            selected_bbox_2d[ii, :] = bbox_2d[cxt_id, view_id, :]
            # selected_bboxsize_2d[ii] = bboxsize_2d[cxt_id, view_id]
            # selected_obj_depth[ii] = obj_depth[cxt_id, view_id]
            selected_campose_2d[ii, :] = campose_2d[cxt_id, view_id, :]

        # if (self.feat2dtype.replace('3D', '')) != 'clsvec':
        #     feat_2d[:len(selected_context_id), :2048] = selected_objfeat_2d  ## ROI feat_2d

        # paste 2D feature
        if self.use_online_visual:
            # construct 2D file list  (unreasonable memory consumption? should try it out!)
            frame_id_to_context = defaultdict(list)
            for jj in range(len(selected_context_id)):  # for each context object
                # cxt_id = selected_context_id[jj]   #
                frame_id = selected_frame_id_2d[jj]
                bbox = selected_bbox_2d[jj]  # xyxy format
                frame_id_to_context[frame_id].append({
                    'order_id': jj,  # we need the order num
                    'bbox': bbox
                })

            for frame_id in frame_id_to_context.keys():
                img_path = os.path.join(self.args.rgb_path, scan.scan_id, 'color', '%06d.jpg' % frame_id)
                im, im_scale = frcn_image_transform(img_path)
                # add res['im'] -> need to check the max frame num for this batch
                # res['im_bbox'], res['im_scale']



            view_images = [os.path.join(self.args.rgb_path, scan.scan_id, 'color', '%06d.jpg' % frame_id)
                           for frame_id in frame_id_to_context.keys()]  # only access selected frames
            for img_path in view_images:



            # img_path = os.path.join(self.args.rgb_path, scan.scan_id, 'color', '%06d.jpg' % frame_id)  # /data/ScanNet/tasks/scannet_frames_25k/scannet_frames_25k, scan.scan_id




        # TODO: move class vector pasting to model.forward
        if self.args.feat2d.startswith('ROI'):
            feat_2d[:len(selected_context_id), :2048] = selected_objfeat_2d
        elif self.args.feat2d.startswith('CLIP'):
            feat_2d[:len(selected_context_id), :768] = selected_objfeat_2d

        if self.args.clsvec2d:  # append one-hot class label
            for ii in range(len(res['class_labels'])):
                if self.args.feat2d.startswith('ROI'):
                    feat_2d[ii, 2048 + res['class_labels'][ii]] = 1.
                elif self.args.feat2d.startswith('CLIP'):
                    feat_2d[ii, 768 + res['class_labels'][ii]] = 1.

        coords_2d[:len(selected_context_id), :] = np.concatenate([selected_bbox_2d, selected_campose_2d[:, :12]],
                                                                 axis=-1)  # bbox + cam_pose
        # coords_2d[:len(selected_context_id),:] = np.concatenate([selected_bbox_2d, selected_obj_depth.reshape(-1,1), selected_campose_2d[:,:12]],axis=-1) ## 1296*968

        # 1296*968 scale down to 0~1
        coords_2d[:, 0], coords_2d[:, 2] = coords_2d[:, 0] / 1296., coords_2d[:, 2] / 1296.
        coords_2d[:, 1], coords_2d[:, 3] = coords_2d[:, 1] / 968., coords_2d[:, 3] / 968.
        # print(selected_ins_id_2d)
        # print(selected_objfeat_2d.shape)
        # # self.max_2d_view
        # print(scan.scan_id,res['class_labels'],samples.shape)
        # print([o.object_id for o in context])
        # print([o.instance_label for o in context])
        # exit(0)
        res['feat_2d'] = feat_2d
        res['coords_2d'] = coords_2d   # bbox + cam_pose
        # res['feat_2d'] = np.random.random(feat_2d.shape).astype(np.float32)
        # res['coords_2d'] = np.random.random(coords_2d.shape).astype(np.float32)

        return res

if __name__ == '__main__':
    dst = "/data/ScanNet/tasks/scannet_frames_25k/scannet_frames_25k/"
    max_len = -1
    for path in tqdm(os.listdir(dst)):
        img_path = os.path.join(dst, path, 'color')
        curr_len = len(os.listdir(img_path))
        print(img_path, curr_len)
        max_len = max(max_len, curr_len)

    print("max_len: ", max_len)
