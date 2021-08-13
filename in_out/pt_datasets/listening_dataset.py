import random
import torch
import time
import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers

from pytorch_transformers.tokenization_bert import BertTokenizer

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from ...data_generation.nr3d import decode_stimulus_string


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, pretrain=False, context_object=False):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.pretrain = pretrain
        self.context_object = context_object

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        # print(scan)
        # print('---------------')
        # print(scan.three_d_objects)
        # print('---------------')
        # print(scan.three_d_objects[ref['target_id']])
        # exit(0)
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

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, text_tokens, is_nr3d = self.get_reference_data(index)
        ## BERT tokenize
        token_inds = torch.zeros(self.max_seq_len, dtype=torch.long)
        indices = self.bert_tokenizer.encode(
            ' '.join(text_tokens), add_special_tokens=True)
        indices = indices[:self.max_seq_len]
        token_inds[:len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)
        if self.pretrain:
            ## entire seq replace for now
            contra_rand=random.random()
            if True:
                tag_pollute = torch.tensor([contra_rand<0.25]).long()
                query_pollute = torch.tensor([contra_rand>0.75]).long()
                contra_pollute = (tag_pollute+query_pollute).clamp(max=1)
            else:
                tag_pollute, query_pollute, contra_pollute = torch.zeros(1).long(),torch.zeros(1).long(),torch.zeros(1).long()
            res['tag_pollute'] = tag_pollute.numpy().astype(np.int64)
            res['query_pollute'] = query_pollute.numpy().astype(np.int64)
            res['contra_pollute'] = contra_pollute.numpy().astype(np.int64)
            ##
            token_inds, mlm_label = self.random_word(token_inds, self.bert_tokenizer.vocab,mask_prob=0.15)

        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])    ## (1024, 6)

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        if self.context_object is not None:
            # ## get context region
            # closest_context, farthest_context, rand_context = [], [], []
            context_obj = []
            for ii in range(len(context)):
                objectA = context[ii]
                dist = np.array([objectA.distance_from_other_object(objectB,optimized=True) for objectB in context])
                dist[dist==0.] = np.mean(dist[dist!=0.]) ## not select as min, max
                # rand = random.choice([i for i in range(len(context)) if i!=ii])
                closest, farthest, rand = int(np.argmin(dist)), int(np.argmax(dist)), random.randint(0,len(dist)-1)
                if self.context_object == 'rand': context_idx = rand
                elif self.context_object == 'closest': context_idx = closest
                elif self.context_object == 'farthest': context_idx = farthest
                pc_context = sample_union_pc(objectA,context[context_idx], scan)
                sampled_idx = np.random.choice(pc_context.shape[0], self.points_per_object, replace=pc_context.shape[0] < self.points_per_object)
                context_obj.append(pc_context[sampled_idx])
            context_obj = np.array(context_obj)
            # closest_context, farthest_context, rand_context = np.array(closest_context), np.array(farthest_context), np.array(rand_context)
            ##
            if self.object_transformation is not None:
                context_obj, context_obj_offset = self.object_transformation(context_obj)
                res['context_offset'] = np.zeros((self.max_context_size, context_obj_offset.shape[1])).astype(np.float32)
                res['context_offset'][:len(context_obj_offset),:] = context_obj_offset.astype(np.float32)
            # take care of padding, so that a batch has same number of N-objects across scans.
            res['context_objects'] = pad_samples(context_obj, self.max_context_size)

        # ## BERT tokenize of class tags
        # print(target.instance_label,[x.instance_label for x in context],len([x.instance_label for x in context]))
        # print(res['class_labels'],len(res['class_labels']))
        # exit(0)
        tag_token_inds = torch.zeros(self.max_context_size, dtype=torch.long)
        tag_indices = self.bert_tokenizer.encode(
            ' '.join([x.instance_label for x in context]), add_special_tokens=True)
        tag_indices = tag_indices[:self.max_context_size]
        tag_token_inds[:len(tag_indices)] = torch.tensor(tag_indices)
        tag_token_num = torch.tensor(len(tag_indices), dtype=torch.long)
        if self.pretrain:
            tag_token_inds, tag_mlm_label = self.random_word(tag_token_inds, self.bert_tokenizer.vocab,mask_prob=0.15)
            mlm_label = torch.cat([mlm_label,tag_mlm_label],dim=0)
        tag_token_inds[0] = 102
        token_inds = torch.cat([token_inds,tag_token_inds],dim=0)
        
        if self.object_transformation is not None:
            # samples = self.object_transformation(samples)
            samples, offset = self.object_transformation(samples)
            res['obj_offset'] = np.zeros((self.max_context_size, offset.shape[1])).astype(np.float32)
            res['obj_offset'][:len(offset),:] = offset.astype(np.float32)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['token_inds'] = token_inds.numpy().astype(np.int64)
        res['token_num'] = token_num.numpy().astype(np.int64)
        res['tag_token_num'] = tag_token_num.numpy().astype(np.int64)
        res['is_nr3d'] = is_nr3d
        if self.pretrain:
            res["mlm_label"] = mlm_label.numpy().astype(np.int64)

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
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

        # ### tmp exp on project map
        # ## (N, 3) (N,) (N, 3)
        # # print(scan.pc.shape,scan.semantic_label.shape,scan.color.shape)
        # print(scan,index)
        # res=128
        # project_img = np.zeros((res,res,3))
        # project_maxh = np.zeros((res,res))
        # max_x, min_x = scan.pc[:,0].max(),scan.pc[:,0].min()
        # max_y, min_y = scan.pc[:,1].max(),scan.pc[:,1].min()
        # scan.pc[:,0] = (scan.pc[:,0]-min_x)/(max_x-min_x+0.01)
        # scan.pc[:,1] = (scan.pc[:,1]-min_y)/(max_y-min_y+0.01)
        # for n in range(scan.pc.shape[0]):
        #     xyz, rgb = scan.pc[n,:], scan.color[n,:]
        #     i,j = int(xyz[0]*res), int(xyz[1]*res)
        #     if project_img[i,j,0]==0 or project_maxh[i,j]<xyz[2]:
        #         project_img[i,j,:] = rgb*255
        #         project_maxh[i,j] = xyz[2]
        # import cv2
        # project_img = cv2.cvtColor(np.float32(project_img), cv2.COLOR_RGB2BGR)
        # cv2.imwrite("%s.jpg"%scan.scan_id, project_img)
        # exit(0)
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
            
            if prob < mask_prob and token not in [0,101,102]:
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
                    token_inds[i] = random.choice(list(range(1000,len(tokenizer))))#[0]

                # -> rest 10% randomly keep current token

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        return token_inds, torch.tensor(output_label)

def sample_union_pc(objectA, objectB, scan):
    extremaA, extremaB = objectA.get_bbox().extrema, objectB.get_bbox().extrema
    union_extrema = np.array([min(extremaA[i],extremaB[i]) for i in range(3)]+[max(extremaA[i],extremaB[i]) for i in range(3,6)],dtype=np.float32)
    # filtered_pc = scan.pc[ (scan.pc[:,0]>union_extrema[0]) & (scan.pc[:,0]<union_extrema[3]) & \
    filtered_pc = np.concatenate([scan.pc,scan.color],axis=1)[ (scan.pc[:,0]>union_extrema[0]) & (scan.pc[:,0]<union_extrema[3]) & \
                            (scan.pc[:,1]>union_extrema[1]) & (scan.pc[:,1]<union_extrema[4]) & \
                            (scan.pc[:,2]>union_extrema[2]) & (scan.pc[:,2]<union_extrema[5]) ]
    return filtered_pc

def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, seed=None):
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

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   pretrain=args.pretrain,
                                   context_object=args.context_obj)

        seed = seed
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
