import random
from collections import defaultdict

import torch
import time
import os
import numpy as np
from torch.utils.data import Dataset, Subset
from functools import partial
from in_out.pt_datasets.utils import dataset_to_dataloader, max_io_workers, extractor_dataset_to_dataloader
from utils import share_array

from pytorch_transformers.tokenization_bert import BertTokenizer

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from in_out.pt_datasets.utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from in_out.pt_datasets.utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from in_out.frcn_read import frcn_image_transform
from data_generation.nr3d import decode_stimulus_string
from PIL import Image
import torchvision.transforms as transforms

from loguru import logger
# from memory_profiler import profile
# import gc
import cv2

# disable cv2 multi-thread processing
cv2.setNumThreads(0)

# class CacheReader:
#     def __getitem__(self, item):
#         return CacheReader._cache_reader()
#     @staticmethod
#     def _cache_reader(scene_name, key_name, cache_path='/dev/shm'):
#         _id = scene_name + '_' + key_name
#         return share_array.read_cache(_id, shm_path=cache_path)

to_tensor_loader = transforms.Compose([transforms.ToTensor()])

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        # lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ListeningDataset(Dataset):
    CLIP_MODEL_INPUT_RESOLUTION = {
        'RN50x16': 384
    }

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
        # self.context_object = context_object  # bool: ????????????context_objects
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
        # self.max_frames = 30

        if self.args.use_clip_visual:  # prepare a preprocessor for clip
            from clip.clip import _transform
            self.clip_img_size = self.CLIP_MODEL_INPUT_RESOLUTION[self.args.clip_backbone]
            self.clip_transform = _transform(self.clip_img_size)

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
            indices = indices[:self.max_seq_len]  # TODO: encode????????????max_seq_len??????????????????EOS??????
            token_inds[:len(indices)] = torch.tensor(indices)
            token_num = torch.tensor(len(indices), dtype=torch.long)
        else:
            clip_indices = self.clip_tokenizer(' '.join(text_tokens))
            clip_indices = clip_indices.squeeze(0)  # have to remove batch_dim, shape: [77] on cpu
            token_num = torch.sum(clip_indices != 0, dtype=torch.long)

        res['csv_index'] = index  # this index varies in train / test split

        if self.use_offline_language:
            res['txt_emb'] = np.load(os.path.join(self.args.offline_language_path,
                                                  '{}_{}_{}.npy'.format(index, self.language_suffix, self.split)))

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

        # if self.context_object is not None:
        #     # ## get context region
        #     # closest_context, farthest_context, rand_context = [], [], []
        #     context_obj = []
        #     for ii in range(len(context)):
        #         objectA = context[ii]
        #         dist = np.array([objectA.distance_from_other_object(objectB, optimized=True) for objectB in context])
        #         dist[dist == 0.] = np.mean(dist[dist != 0.])  ## not select as min, max
        #         # rand = random.choice([i for i in range(len(context)) if i!=ii])
        #         closest, farthest, rand = int(np.argmin(dist)), int(np.argmax(dist)), random.randint(0, len(dist) - 1)
        #         if self.context_object == 'rand':
        #             context_idx = rand
        #         elif self.context_object == 'closest':
        #             context_idx = closest
        #         elif self.context_object == 'farthest':
        #             context_idx = farthest
        #         pc_context = sample_union_pc(objectA, context[context_idx], scan)
        #         sampled_idx = np.random.choice(pc_context.shape[0], self.points_per_object,
        #                                        replace=pc_context.shape[0] < self.points_per_object)
        #         context_obj.append(pc_context[sampled_idx])
        #     context_obj = np.array(context_obj)
        #     # closest_context, farthest_context, rand_context = np.array(closest_context), np.array(farthest_context), np.array(rand_context)
        #     ##
        #     if self.object_transformation is not None:
        #         context_obj, context_obj_offset = self.object_transformation(context_obj)
        #         res['context_offset'] = np.zeros((self.max_context_size, context_obj_offset.shape[1])).astype(
        #             np.float32)
        #         res['context_offset'][:len(context_obj_offset), :] = context_obj_offset.astype(np.float32)
        #     # take care of padding, so that a batch has same number of N-objects across scans.
        #     res['context_objects'] = pad_samples(context_obj, self.max_context_size)

        #########################################
        # ## BERT tokenize of class tags
        # ## V0: prev enc
        # tag_token_inds = torch.zeros(self.max_context_size, dtype=torch.long)
        # tag_indices = self.bert_tokenizer.encode(
        #     ' '.join([x.instance_label for x in context]), add_special_tokens=True)
        # tag_indices = tag_indices[:self.max_context_size]

        # ## V1: updated enc (first word label name only); tmp approx for now; TODO
        # tag_token_inds = torch.zeros(self.max_context_size+2, dtype=torch.long) ## CLS, SEP
        # tag_indices = [self.bert_tokenizer.encode(x.instance_label, add_special_tokens=False)[0] for x in context]
        # tag_indices = [101] + tag_indices + [102]
        # assert(len(tag_indices)==(len(context)+2))
        # tag_indices = tag_indices[:self.max_context_size+2]

        # if self.addlabel_words:
        #     ## V2: 12/23; option 1: add all words in obj cls in dictionary (include spaces)
        #     ## option 2: record the word split, and merge the feature later accordingly
        #     ## Implementation of option 2
        #     tag_token_inds = torch.zeros(self.max_context_size + 2, dtype=torch.long)  ## CLS, SEP
        #     word_split_inds = torch.ones(self.max_context_size,
        #                                  dtype=torch.long) * -1.  ## mismatch by 1 position with tag token
        #     # tag_indices_list = [self.bert_tokenizer.encode(x.instance_label, add_special_tokens=False) for x in context]
        #     tag_indices_list = [[self.bert_tokenizer.encode(x.instance_label, add_special_tokens=False)[0]] for x in
        #                         context]
        #     word_split_idx, tag_indices = [], []
        #     for ii in range(len(tag_indices_list)):
        #         # word_split_idx += [ii for x in range(len(tag_indices_list[ii]))]
        #         word_split_idx += [ii] * len(tag_indices_list[ii])
        #         tag_indices += tag_indices_list[ii]
        #     tag_indices = [101] + tag_indices + [102]
        #     tag_indices = tag_indices[:self.max_context_size + 2]
        #     word_split_idx = word_split_idx[:self.max_context_size]
        #     word_split_inds[:len(word_split_idx)] = torch.tensor(word_split_idx)
        #     res['word_split_inds'] = word_split_inds.numpy().astype(np.int64)
        #     ## end of enc
        #     #########################################
        #
        #     tag_token_inds[:len(tag_indices)] = torch.tensor(tag_indices)
        #     tag_token_num = torch.tensor(len(tag_indices), dtype=torch.long)
        #     if self.pretrain:
        #         tag_token_inds, tag_mlm_label = self.random_word(tag_token_inds, self.bert_tokenizer.vocab,
        #                                                          mask_prob=0.15)
        #         mlm_label = torch.cat([mlm_label, tag_mlm_label], dim=0)
        #     tag_token_inds[0] = 102
        #     token_inds = torch.cat([token_inds, tag_token_inds], dim=0)

        if self.object_transformation is not None:
            # samples = self.object_transformation(samples)
            samples, offset = self.object_transformation(samples)
            res['obj_offset'] = np.zeros((self.max_context_size, offset.shape[1])).astype(np.float32)
            res['obj_offset'][:len(offset), :] = offset.astype(np.float32)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)  # ??????????????????1

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        # # with open('idx_525.txt','w') as f:
        # with open('idx_608.txt','w') as f:
        #     for k in self.class_to_idx:
        #         i=self.class_to_idx[k]
        #         bert_idx = self.bert_tokenizer.encode(k, add_special_tokens=False)[0]
        #         print(k,i,bert_idx)
        #         f.write('%d,%d\n'%(i,bert_idx))
        # exit(0)

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask  # indicating which objects have the same instance-class as the target.

        if not self.args.use_clip_language:
            res['tokens'] = tokens
            res['token_inds'] = token_inds.numpy().astype(np.int64)  # model takes these
            # res['token_num'] = token_num.numpy().astype(np.int64)
        else:
            res['clip_inds'] = clip_indices

        res['token_num'] = token_num.numpy().astype(np.int64)
        # if self.addlabel_words: res['tag_token_num'] = tag_token_num.numpy().astype(np.int64)
        res['is_nr3d'] = is_nr3d
        # if self.pretrain:
        #     res["mlm_label"] = mlm_label.numpy().astype(np.int64)

        if self.visualization:
            distrators_pos = np.zeros(6)  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros(self.max_context_size)
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id
            # object_size = open('object_size.txt','a')
            # object_size.write('%s,%d\n'%(res['stimulus_id'],len(context[target_pos].points)))
            # object_size.close()
        if self.evalmode:  # strictly enforce no 2D context leak in eval mode
            if self.args.feat2d.startswith('ROI'):
                featdim = 2048
            elif self.args.feat2d.startswith('CLIP_add'):
                featdim = 768
            elif self.args.feat2d.startswith('CLIP'):
                featdim = 768
            else:
                raise NotImplemented()
            if self.args.clsvec2d:
                featdim += self.num_class_dim
            feat_2d = np.zeros((self.max_context_size, featdim)).astype(np.float32)
            coords_2d = np.zeros((self.max_context_size, 4 + 12)).astype(
                np.float32)  # make empty feat_2d & coords_2d, since they do not matter in eval
            res['feat_2d'] = feat_2d
            res['coords_2d'] = coords_2d
            return res

        # TODO: read in image patches if use_clip_visual is enabled, ????????????2
        # objfeat_2d = False if self.use_online_visual else True  # set objfeat_2d to False if planning to use online RoI
        # TODO: ?????????obj_feat2d???????????????
        objfeat_2d = True

        if not self.args.offline_cache:  # no caching...
            # context_2d = np.load('%s/%s.npy' % (self.offline_2d_feat, scan.scan_id),
            #                      allow_pickle=True, encoding='latin1')
            # choose objfeat_2d based on args.feat2d
            if self.args.feat2d.startswith('ROI'):
                objfeat_2d = self._load_split_offline_npy(scan.scan_id, 'obj_feat') if objfeat_2d else False
                featdim = 2048
            elif self.args.feat2d.startswith('CLIP_add'):
                objfeat_2d = self._load_split_offline_npy(scan.scan_id, 'clip_region_feat') + \
                             self._load_split_offline_npy(scan.scan_id,
                                                          'clip_scaled_region_feat') if objfeat_2d else False
                featdim = 768
            elif self.args.feat2d.startswith('CLIP'):  # feat that do not norm!! should consider?
                objfeat_2d = self._load_split_offline_npy(scan.scan_id,
                                                          'clip_region_feat') if objfeat_2d else False  # also do not norm...
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
                objfeat_2d = self._cache_reader(scan.scan_id,
                                                'obj_feat') if objfeat_2d else False  # also do not norm...
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

        # if self.use_online_visual:  # if not loading offline features, we provide loaded images and bbox for pretrained Faster-RCNN
        #     feat_2d = None
        # else:
        feat_2d = np.zeros((self.max_context_size, featdim)).astype(np.float32)

        coords_2d = np.zeros((self.max_context_size, 4 + 12)).astype(np.float32)
        # coords_2d = np.zeros((self.max_context_size, 4+1+12)).astype(np.float32)

        selected_2d_idx = 0
        # selected_2d_idx = [random.randint(0, max(0,int((ins_id_2d[ii,:]!=0).astype(np.float32).sum())-1)) for ii in range(ins_id_2d.shape[0])]
        #
        selected_context_id = [o.object_id + 1 for o in context]  # background included, so +1
        # print(scan.scan_id,objfeat_2d.shape,selected_context_id)

        # first choose, all view-0 on dim 1
        # if self.use_online_visual:
        #     selected_objfeat_2d = None
        # else:
        selected_objfeat_2d = objfeat_2d[selected_context_id, selected_2d_idx, :]  ## ROI feat_2d

        selected_bbox_2d = bbox_2d[selected_context_id, selected_2d_idx, :]
        # selected_bboxsize_2d = bboxsize_2d[selected_context_id, selected_2d_idx]
        # selected_obj_depth = obj_depth[selected_context_id, selected_2d_idx]
        selected_campose_2d = campose_2d[selected_context_id, selected_2d_idx, :]
        # selected_ins_id_2d = ins_id_2d[selected_context_id, selected_2d_idx]
        selected_frame_id_2d = frame_id_2d[selected_context_id, selected_2d_idx]

        # if True:  # use random selected_2d_idx, instead of 0 (dummy if True, removed)
        # context_to_view = dict()  # ???????????????context???????????????view (?????????)

        for ii in range(len(selected_context_id)):
            cxt_id = selected_context_id[ii]
            view_id = random.randint(0, max(0, int((ins_id_2d[cxt_id, :] != 0).astype(
                np.float32).sum()) - 1))  # ?????????context object ?????????????????????view???view_id??????index???frame???????????????
            # ??????????????????????????????(100 frames -> view)????????????????????????3D Object???????????????????????????2D image???
            # context_to_view[cxt_id] = view_id
            # if self.use_online_visual:
            selected_frame_id_2d[ii] = frame_id_2d[cxt_id, view_id]  # need frame id to reference RGB image
            # else:
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
                bbox = selected_bbox_2d[jj]  # xyxy format, np array
                # add zero bbox checker
                if np.all(bbox == 0) or np.abs(bbox[2] - bbox[0]) < 5 or np.abs(bbox[3] - bbox[1]) < 5:  # skip empty / unreasonable 2D object
                    continue
                # frame_id_to_context[frame_id].append({
                #     'order_id': jj,  # order_id used to indicate bbox<->object (whether it is a target)
                #     'bbox': bbox
                # })
                frame_id_to_context[frame_id].append((jj, bbox))

            if self.args.use_frcn_visual:
                im_list, im_scale_list = [], []
                # we have to self-define a collate_fn, to allow size-mutable frame_list...
                im_to_obj = []  # index 0: [obj_list, bbox_list]

                for frame_id in frame_id_to_context.keys():  # for each selected frame
                    img_path = os.path.join(self.args.rgb_path, scan.scan_id, 'color', '%06d.jpg' % frame_id)
                    im, im_scale = frcn_image_transform(img_path)
                    # res['im'], res['im_scale'], res['im_to_obj']
                    im_list.append(im)
                    im_scale_list.append(im_scale)
                    im_to_obj.append(frame_id_to_context[frame_id])

                res['im'] = im_list
                res['im_scale'] = im_scale_list
                res['im_to_obj'] = im_to_obj  # [[(4, bbox1), (7, bbox2)](?????????frame??????obj,bbox??????), [], ...]
            elif self.args.use_clip_visual:
                clip_img_data = np.zeros((self.max_context_size, 3, self.clip_img_size, self.clip_img_size),
                                         dtype=np.float32)  # store img data, some of object image will be zero
                for frame_id in frame_id_to_context.keys():  # for each selected frame
                    img_path = os.path.join(self.args.rgb_path, scan.scan_id, 'color', '%06d.jpg' % frame_id)
                    img = Image.open(img_path, 'r').convert('RGB')  # full img here (C, H, W)
                    for obj, bbox in frame_id_to_context[frame_id]:
                        x1, y1, x2, y2 = bbox.astype(np.int)
                        # region_crop = img[:, y1:y2, x1:x2]
                        region_crop = img.crop((x1, y1, x2, y2))
                        region_crop = self.clip_transform(region_crop)  # (3, clip_size, clip_size)
                        clip_img_data[obj] = region_crop  # imexplict convesion from tensor to nparray
                res['clip_crops'] = clip_img_data  # (max_context_size, 3, clip_size, clip_size)

            else:
                raise NotImplemented()

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
                                                                 axis=-1)  # bbox + cam_pose(3x4)
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
        res['feat_2d'] = feat_2d  # always need this, whether online or offline
        res['coords_2d'] = coords_2d  # bbox + cam_pose
        # res['feat_2d'] = np.random.random(feat_2d.shape).astype(np.float32)
        # res['coords_2d'] = np.random.random(coords_2d.shape).astype(np.float32)

        return res

    """
    From VilBert, dataset/concept_cap_dataset
    """

    def random_word(self, token_inds, tokenizer, mask_prob=0.15):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        assert tokenizer['[MASK]'] == 103
        assert tokenizer['[UNK]'] == 100
        mask_prob = mask_prob

        output_label = []
        for i, token in enumerate(token_inds):
            prob = random.random()
            # mask token with 15% probability

            if prob < mask_prob and token not in [0, 101, 102]:
                # append current token to output (we will predict these later)
                try:
                    # output_label.append(tokenizer.vocab[token])
                    output_label.append(int(token))
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    # output_label.append(tokenizer.vocab["[UNK]"])
                    output_label.append(100)
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )

                prob /= mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # token_inds[i] = "[MASK]"
                    token_inds[i] = 103

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # token_inds[i] = random.choice(list(tokenizer.vocab.items()))[0]
                    token_inds[i] = random.choice(list(range(1000, len(tokenizer))))  # [0]

                # -> rest 10% randomly keep current token

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        return token_inds, torch.tensor(output_label)


def sample_union_pc(objectA, objectB, scan):
    extremaA, extremaB = objectA.get_bbox().extrema, objectB.get_bbox().extrema
    union_extrema = np.array(
        [min(extremaA[i], extremaB[i]) for i in range(3)] + [max(extremaA[i], extremaB[i]) for i in range(3, 6)],
        dtype=np.float32)
    # filtered_pc = scan.pc[ (scan.pc[:,0]>union_extrema[0]) & (scan.pc[:,0]<union_extrema[3]) & \
    filtered_pc = np.concatenate([scan.pc, scan.color], axis=1)[
        (scan.pc[:, 0] > union_extrema[0]) & (scan.pc[:, 0] < union_extrema[3]) & \
        (scan.pc[:, 1] > union_extrema[1]) & (scan.pc[:, 1] < union_extrema[4]) & \
        (scan.pc[:, 2] > union_extrema[2]) & (scan.pc[:, 2] < union_extrema[5])]
    return filtered_pc


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, seed=None, cut_prefix_num=None):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        #
        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(args=args,
                                   references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   offline_2d_feat=args.offline_2d_feat,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   pretrain=args.pretrain,
                                   context_object=args.context_obj,
                                   feat2dtype=args.feat2d,
                                   addlabel_words=args.addlabel_words,
                                   num_class_dim=525 if '00' in args.scannet_file else 608,
                                   # ????????????????????????Tips: Nr3D???Sr3D???type????????????
                                   evalmode=(args.mode == 'evaluate'),
                                   split=split)  # ??????????????????eval????????????

        if split == 'train' and cut_prefix_num is not None:  # split subset form profiling
            dataset = Subset(dataset, np.arange(cut_prefix_num))
            n_workers = 0
            logger.info("Slicing the prev {} samples for train-set profiling!".format(cut_prefix_num))

        seed = seed
        if split == 'test':
            seed = args.random_seed

        # not_stacked_keys = ['im', 'im_scale'] if args.rgb_path != "" else None\
        not_stacked_keys = None  # TODO: FRCN Training impossible for now
        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, pin_memory=False,
                                                    seed=seed, not_stacked_keys=not_stacked_keys)

    return data_loaders


def make_extractor_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, seed=None,
                                cut_prefix_num=None):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        #
        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.
        # TODO: why not keep the same?

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(args=args,
                                   references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   offline_2d_feat=args.offline_2d_feat,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   pretrain=args.pretrain,
                                   context_object=args.context_obj,
                                   feat2dtype=args.feat2d,
                                   addlabel_words=args.addlabel_words,
                                   num_class_dim=525 if '00' in args.scannet_file else 608,
                                   # ????????????????????????Tips: Nr3D???Sr3D???type????????????
                                   evalmode=(args.mode == 'evaluate'))  # ??????????????????eval???????????? ????????????eval????????????2D??????????????????

        if split == 'train' and cut_prefix_num is not None:  # split subset form profiling
            dataset = Subset(dataset, np.arange(cut_prefix_num))
            n_workers = 0
            logger.info("Slicing the prev {} samples for train-set profiling!".format(cut_prefix_num))

        seed = seed
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = extractor_dataset_to_dataloader(dataset, split, args.batch_size, n_workers,
                                                              pin_memory=False,
                                                              seed=seed)

    return data_loaders
