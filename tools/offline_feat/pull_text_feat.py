"""
pull out offline text feature

usage (clip text encoder):
CUDA_VISIBLE_DEVICES=3 python3 split_feat.py --batch-size=36 --gpu=3 --transformer --experiment-tag=clip_lang_eos \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
--clip-backbone RN50x16 --use-clip-language \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--norm-offline-feat --n-workers 4 --extract-text

usage (text bert encoder):
CUDA_VISIBLE_DEVICES=3 python3 split_feat.py --batch-size=36 --gpu=3 --transformer --experiment-tag=clip_lang_eos \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--norm-offline-feat --n-workers 4 --extract-text
"""
from collections import defaultdict

import os

import numpy as np
import torch
from in_out.arguments import parse_arguments
from in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from in_out.neural_net_oriented import load_scan_related_data, load_referential_data
# from in_out.pt_datasets.listening_dataset import make_data_loaders
from in_out.pt_datasets.listening_dataset_2dcontext import make_data_loaders, make_extractor_data_loaders

from models.sat_net import instantiate_referit3d_net
from utils import set_gpu_to_zero_position, seed_training_code, create_logger
from utils.scheduler import GradualWarmupScheduler
from utils.tf_visualizer import Visualizer
from tqdm import tqdm
import pandas as pd

# DST_PATH = '/data/meta-ScanNet/nr3d_bert_text'  # bert feature here
# DST_SUFFIX = 'bert_text_feat'  # example npy file name: 0_bert_text_feat_train.npy

DST_PATH = '/data/meta-ScanNet/nr3d_clip_text'  # clip_lang_eos feature here
DST_SUFFIX = 'clip_text_feat'  # example npy file name: 0_clip_text_feat_train.npy
ARGS = None


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    """
    0在PAD住的位置上，在max_num的seq上不mask前nums个element
    """
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@torch.no_grad()
def extract_text_feat(model, batch, is_train):
    """
    this function only saves a [N, lang_size, d_feat] language feature.
    no classifier used!
    Use with cautions! language order might be shuffled...
    """
    # result = defaultdict(lambda: None)
    for k in batch.keys():
        batch[k] = batch[k].to(device)

    if not ARGS.use_clip_language:
        txt_inds = batch["token_inds"]  # N, lang_size  lang_size = args.max_seq_len
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.
        txt_mask = _get_mask(batch['token_num'].to(txt_inds.device),  # how many token are not masked
                             txt_inds.size(1))  ## all proposals are non-empty
        txt_type_mask = txt_type_mask.long()

        text_bert_out = model.text_bert(
            txt_inds=txt_inds,
            txt_mask=txt_mask,
            txt_type_mask=txt_type_mask
        )  # N, lang_size, TEXT_BERT_HIDDEN_SIZE
        txt_emb = model.text_bert_out_linear(text_bert_out)  # text_bert_hidden_size -> mmt_hidden_size
        # Classify the target instance label based on the text
        # if model.language_clf is not None:
        #     result['lang_logits'] = model.language_clf(
        #         text_bert_out[:, 0, :])  # language classifier only use [CLS] token
    else:  # clip language encoder
        txt_inds = batch['clip_inds']
        txt_emb = model.clip_model.encode_text(txt_inds)  # txt embeddings shape: [N, lang_size, 768]
        # txt_mask = _get_mask(batch['token_num'].to(batch['clip_inds'].device),  # how many token are not masked
        #                      batch['clip_inds'].size(1))
        # if model.language_clf is not None:
        #     # result['lang_logits'] = self.language_clf(txt_emb[:, 0, :])
        #     # !! BUG Found !! txt_emb[:, 0, :] will always be the same in clip encoder
        #     txt_cls_emb = model.clip_model.classify_text(batch['clip_inds'],
        #                                                  txt_emb)  # N, 768  -> Direct EOS works better
        #     result['lang_logits'] = model.language_clf(txt_cls_emb)
    # print(txt_emb.shape)
    for j in range(txt_emb.size(0)):  # iterate over each utterance
        index = batch['csv_index'][j]
        save_path = os.path.join(DST_PATH, '{}_{}_{}.npy'.format(index, DST_SUFFIX, 'train' if is_train else 'test'))
        save_obj = txt_emb[j].detach().cpu().numpy()
        # save_obj = {'data': txt_emb[j].detach().cpu().numpy(), 'inds': txt_inds[j].cpu().numpy(), 'index': int(index)}
        # np.save(save_path, save_obj, allow_pickle=True)  # save a [lang_size, d_model] array
        np.save(save_path, save_obj)


if __name__ == '__main__':
    args = parse_arguments()
    # exit(0)
    ARGS = args
    if not os.path.exists(DST_PATH):
        os.makedirs(DST_PATH)

    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0", make only args.gpu visible by torch
    device = torch.device('cuda')
    # TODO: now torch seems ignore in-place env setting, using CUDA_VISIBLE_DEVICES instead...

    # if args.wandb_log:
    #     wandb_init(args)
    #
    # if args.git_commit:
    #     save_code_to_git('-'.join(args.log_dir.split('/')[-2:]))  # extract the last two parts of args.logdir

    logger = create_logger(args.log_dir)  # setting a global logger

    if args.context_2d != 'unaligned':
        args.mmt_mask = None
        logger.info('not in unaligned mode, set mmt-mask to None!\n')

    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)  # nr3d.csv/sr3d.csv就ok

    # referit_data.to_csv('filter_nr3d.csv')
    # exit(0)
    # dump a referit_data to check

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_extractor_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb,
                                               cut_prefix_num=1000 if args.profile or args.debug else None)

    # Prepare GPU environment
    # set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0", make only args.gpu visible by torch
    # device = torch.device('cuda')
    # device = torch.device('cuda:' + str(args.gpu))
    torch.backends.cudnn.benchmark = True
    seed_training_code(args.random_seed, strict=True)  # should speed up using cudnn

    n_classes = len(class_to_idx) - 1
    model = instantiate_referit3d_net(args, vocab, n_classes).to(device)

    model.eval()

    # data_loaders['train'].drop_last = data_loaders['test'].drop_last = False
    # data_loaders['train'].batch_size = data_loaders['test'].batch_size = 1

    print(data_loaders['train'].dataset.__len__())  # 28716
    print(data_loaders['test'].dataset.__len__())  # 7485
    # exit(0)

    for batch in tqdm(data_loaders['train']):
        extract_text_feat(model, batch, is_train=True)

    for batch in tqdm(data_loaders['test']):
        extract_text_feat(model, batch, is_train=False)
