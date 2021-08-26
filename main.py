"""
Refactored Train Script for SAT and its variants
Usage: (TBD)

师兄给的例子：
python train_py
--init-lr $lr
--batch-size $bs --transformer
--model mmt_referIt3DNet
-scannet-file $scanfile
-referit3D-file $nr3dfile
--log-dir log/$exp_id
--unit-sphere-norm True
--feat2d $feat2d
--context_2d $context_2d
--mmt_mask $mmt_mask
--warmup

e.g.: (ROI feat) 2021-08-18 09:55:29 --  roi_feat/08-16-2021-19-38-29  复现结果 47.5
python main.py --init-lr 0.0001 --batch-size=36 --gpu=0 --transformer --experiment-tag=roi_feat \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d ROI --clsvec2d --context_2d unaligned --mmt_mask train2d --save-args --n-workers 4 --wandb-log --git-commit

(ROI feat replicated)   0.480 (@epoch 93)
python main.py --init-lr 0.0001 --batch-size=16 --gpu=3 --transformer --experiment-tag=roi_feat \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d ROI --clsvec2d --context_2d unaligned --mmt_mask train2d --save-args --n-workers 4 --wandb-log --git-commit

(ROI feat replicated evaluate)
CUDA_VISIBLE_DEVICES=2 python main.py --mode evaluate --init-lr 0.0001 --batch-size=64 --gpu=2 --transformer --experiment-tag=roi_feat \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
--resume-path /data/logs/roi_feat/08-17-2021-09-39-20/checkpoints/best_model.pth -offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --unit-sphere-norm True \
--feat2d ROI --clsvec2d --context_2d unaligned --mmt_mask train2d --n-workers 4 --analyze-evaluate

(ROI evaluate) 47.6+-0.2
CUDA_VISIBLE_DEVICES=2 python main.py --mode evaluate --init-lr 0.0001 --batch-size=64 --gpu=2 --transformer --experiment-tag=roi_feat \
--model mmt_referIt3DNet -scannet-file /dev/shm/keep_all_points_00_view_with_global_scan_alignment.pkl \
--resume-path /data/logs/roi_feat/08-16-2021-19-38-29/checkpoints/best_model.pth -offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --unit-sphere-norm True \
--feat2d ROI --clsvec2d --context_2d unaligned --mmt_mask train2d --n-workers 4 --analyze-evaluate

# feat2d不再重要，只为了对齐shape

e.g. 2: (CLIP_add feat)   Done: clip_add/08-16-2021-19-55-34   复现结果46.6
python main.py --init-lr 0.0001 --batch-size=36 --gpu=1 --transformer --experiment-tag=clip_add \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d --save-args --n-workers 4 --wandb-log --git-commit

e.g. 3: (CLIP_add norm feat)   clip_add_norm/08-18-2021-10-01-32  0.476 (@epoch 95)
CUDA_VISIBLE_DEVICES=1 python main.py --init-lr 0.0001 --batch-size=36 --gpu=1 --transformer --experiment-tag=clip_add_norm \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--save-args --norm-offline-feat --n-workers 4 --wandb-log --git-commit

(CLIP_add norm evaluate)  47.4 (0.2)
CUDA_VISIBLE_DEVICES=2 python main.py --mode evaluate --init-lr 0.0001 --batch-size=64 --gpu=2 --transformer --experiment-tag=roi_feat \
--model mmt_referIt3DNet -scannet-file /dev/shm/keep_all_points_00_view_with_global_scan_alignment.pkl \
--resume-path /data/logs/clip_add/08-16-2021-19-55-34/checkpoints/best_model.pth -offline-2d-feat /data/meta-ScanNet/split_feat/ \
-referit3D-file /data/meta-ScanNet/nr3d.csv --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d --n-workers 4 --analyze-evaluate


e.g. 4: (CLIP_add norm offline feat & CLIP language online encoder) 有问题，train_txt_cls_acc一直不涨
CUDA_VISIBLE_DEVICES=1 python main.py --init-lr 0.0001 --batch-size=24 --gpu=1 --transformer --experiment-tag=clip_lang \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
--clip-backbone RN50x16 --use-clip-language \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--save-args --norm-offline-feat --n-workers 4 --wandb-log --git-commit

(重置CLIP)  clip_lang/08-26-2021-12-43-11
CUDA_VISIBLE_DEVICES=1 python main.py --init-lr 0.0001 --batch-size=24 --gpu=1 --transformer --experiment-tag=clip_lang \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
--clip-backbone RN50x16 --use-clip-language \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--save-args --norm-offline-feat --n-workers 4 --init-language

(之前ok的clip-add-norm base，但重置一下textbert，看看bert预训练的消融 -- 分析原因的实验)  clip_add_norm/08-26-2021-16-41-50
CUDA_VISIBLE_DEVICES=0 python main.py --init-lr 0.0001 --batch-size=36 --gpu=0 --transformer --experiment-tag=clip_add_norm \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ --init-language \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--save-args --norm-offline-feat --n-workers 4 --wandb-log --git-commit

(只用EOS不做text_projection的CLIP-direct)
CUDA_VISIBLE_DEVICES=2 python main.py --init-lr 0.0001 --batch-size=26 --gpu=2 --transformer --experiment-tag=clip_lang_eos \
--model mmt_referIt3DNet -scannet-file /data/meta-ScanNet/pkl_nr3d/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
-offline-2d-feat /data/meta-ScanNet/split_feat/ \
--clip-backbone RN50x16 --use-clip-language \
-referit3D-file /data/meta-ScanNet/nr3d.csv --log-dir /data/logs/ --unit-sphere-norm True \
--feat2d CLIP_add --clsvec2d --context_2d unaligned --mmt_mask train2d \
--save-args --norm-offline-feat --n-workers 4 --direct-eos


"""

import time
import torch
import torch.nn as nn
# from loguru import logger
from torch import optim
from termcolor import colored
import tqdm
import os.path as osp

from analysis.deepnet_predictions import analyze_predictions
from models.sat_net_utils import evaluate_on_dataset, single_epoch_train, single_epoch_debug
from models.utils import load_state_dicts, save_state_dicts
from in_out.arguments import parse_arguments
from in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from in_out.neural_net_oriented import load_scan_related_data, load_referential_data
# from in_out.pt_datasets.listening_dataset import make_data_loaders
from in_out.pt_datasets.listening_dataset_2dcontext import make_data_loaders
# from in_out.pt_datasets.listening_dataset_2dcontext_numpy import make_data_loaders    # will this solve issue?

from models.sat_net import instantiate_referit3d_net
from utils import set_gpu_to_zero_position, seed_training_code, create_logger, wandb_init, save_code_to_git
from utils.scheduler import GradualWarmupScheduler
from utils.tf_visualizer import Visualizer

if __name__ == '__main__':
    def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])

            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


    # Parse arguments
    args = parse_arguments()

    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0", make only args.gpu visible by torch
    device = torch.device('cuda')
    # TODO: now torch seems ignore in-place env setting, using CUDA_VISIBLE_DEVICES instead...

    if args.wandb_log:
        wandb_init(args)

    if args.git_commit:
        save_code_to_git('-'.join(args.log_dir.split('/')[-2:]))   # extract the last two parts of args.logdir

    logger = create_logger(args.log_dir)  # setting a global logger

    if args.context_2d != 'unaligned':
        args.mmt_mask = None
        logger.info('not in unaligned mode, set mmt-mask to None!\n')

    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)  # nr3d.csv/sr3d.csv就ok

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb,
                                     cut_prefix_num=1000 if args.profile or args.debug else None)

    # Prepare GPU environment
    # set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0", make only args.gpu visible by torch
    # device = torch.device('cuda')
    # device = torch.device('cuda:' + str(args.gpu))
    torch.backends.cudnn.benchmark = True
    seed_training_code(args.random_seed, strict=False)  # should speed up using cudnn

    # Losses:
    criteria = dict()

    # Referential, "find the object in the scan" loss
    if args.s_vs_n_weight is not None:  # joint training coefficient weight
        assert args.augment_with_sr3d is not None
        ce = nn.CrossEntropyLoss(reduction='none').to(device)
        s_vs_n_weight = args.s_vs_n_weight


        def weighted_ce(logits, batch):
            loss_per_example = ce(logits, batch['target_pos'])
            sr3d_mask = ~batch['is_nr3d']
            loss_per_example[sr3d_mask] *= s_vs_n_weight
            loss = loss_per_example.sum() / len(loss_per_example)
            return loss


        criteria['logits'] = weighted_ce
    else:
        criteria['logits'] = nn.CrossEntropyLoss().to(device)  # logits全部用CE Loss

    # if args.obj_cls_alpha > 0:  # obj-type classification
    #     criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad']).to(device)
    #
    # # Target-in-language guessing
    # if args.lang_cls_alpha > 0:
    #     criteria['lang_logits'] = nn.CrossEntropyLoss().to(device)
    criteria['logits_nondec'] = nn.CrossEntropyLoss(reduction='none').to(device)

    # Object-type classification
    if args.obj_cls_alpha > 0:
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad'], reduction=reduction).to(device)
        # criteria['class_logits'] = FocalLoss(gamma=2,ignore_index=class_to_idx['pad']).to(device)

    # Target-in-language guessing
    if args.lang_cls_alpha > 0:
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        criteria['lang_logits'] = nn.CrossEntropyLoss(reduction=reduction).to(device)
        # criteria['lang_logits'] = FocalLoss(gamma=2).to(device)

    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']

    model = instantiate_referit3d_net(args, vocab, n_classes).to(device)  # n_classes: number of instance labels
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
    #                                                           patience=5, verbose=True)

    # make sure backbone holds 1/10 lr
    same_backbone_lr = False
    if same_backbone_lr:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    else:
        backbone_name = []
        if args.transformer:
            backbone_name.append('text_bert.')  ## exclude text_bert_out_linear
            # backbone_name.append('object_encoder.')
            # backbone_name.append('cnt_object_encoder.')
        if args.use_clip_visual:
            backbone_name.append('clip_model.visual.')

        if args.use_clip_language:
            backbone_name.append('clip_model.transformer.')

        backbone_param, rest_param = [], []
        for kv in model.named_parameters():
            isbackbone = [int(key in kv[0]) for key in backbone_name]
            if sum(isbackbone + [0]):
                backbone_param.append(kv[1])
            else:
                rest_param.append(kv[1])
        optimizer = optim.Adam([{'params': rest_param},
                                {'params': backbone_param, 'lr': args.init_lr / 10.}],
                               lr=args.init_lr)  # text_bert -> 1/10 LR

        sum_backbone = sum([param.nelement() for param in backbone_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        sum_all = sum([param.nelement() for param in model.parameters()])
        logger.info('backbone, fusion module parameters: {}, {}, {}'.format(sum_backbone, sum_fusion, sum_all))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
                                                              patience=10 if args.patience >= 30 else 5, verbose=True)
    # patience=5, verbose=True)
    if args.patience == args.max_train_epochs:  # will triggered in default params settings
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[23,37,43,51,60,71,79,87], gamma=0.65)    ## custom1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40, 50, 60, 70, 80, 90],
                                                            gamma=0.65)  # in our setting
        # if args.max_train_epochs == 120:
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                         milestones=[25, 40, 55, 70,
        #                                                                     85, 100],
        #                                                         gamma=0.5)  ## custom3-120ep
    if args.warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)
        optimizer.zero_grad()  # this zero gradient update is needed to avoid a warning message, issue #8.
        optimizer.step()

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    no_improvement = 0

    if args.resume_path:
        logger.warning('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch, best_test_acc = load_state_dicts(args.resume_path, map_location=device, model=model)
        logger.info('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if args.mode == 'train':
            if not args.fine_tune:  # just resume, not fine-tune, load optimizer & scheduler
                logger.info('Loaded a model that we do NOT plan to fine-tune.')
                load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
                start_training_epoch = loaded_epoch + 1
                best_test_epoch = loaded_epoch
                # best_test_acc = lr_scheduler.best
                if best_test_acc is not None:
                    logger.info('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                        best_test_acc))
                else:
                    best_test_acc = -1  # to be compatible with older version
            else:
                logger.info('Parameters that do not allow gradients to be back-propped:')
                ft_everything = True
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        logger.info(name)
                        exist = False
                if ft_everything:
                    logger.info('None, all will be fine-tuned')
                # if you fine-tune the previous epochs/accuracy are irrelevant.
                dummy = args.max_train_epochs + 1 - start_training_epoch
                logger.info('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))
        else:
            logger.info('Loaded model parameters, ready for evaluating...')

    if args.pretrain_path:  # for evaluating abandoned
        load_model = torch.load(args.pretrain_path)
        pretrained_dict = load_model['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()]) != 0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info("=> loaded pretrain model at {}"
                    .format(args.pretrain_path))
        if 'best_test_acc' in load_model:
            logger.info('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                load_model['best_test_acc']))

    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir, use_wandb=args.wandb_log)
        logger.info('Starting the training. Good luck!')
        eval_acc = 0.

        if args.profile:
            logger.info('Starting Profiling...')
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
            #                                      profile_memory=False) as prof:
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('/data/logs/profile'),
                    record_shapes=True,
                    with_stack=True
            ) as prof:
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args, epoch=0)  # train 1 epoch for profiling
                # print(prof.table())
            # prof.export_chrome_trace('./sat_train_profile.json')
            exit(0)

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                # Train:
                if args.warmup:
                    print('warmup triggered...')
                    scheduler_warmup.step(epoch=epoch, metrics=eval_acc)  # using the previous epoch's metrics
                logger.info('Current LR (epoch {}): rest {}, backbone {}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                                 optimizer.param_groups[1]['lr']))

                tic = time.time()
                if not args.debug:
                    if not args.skip_train:
                        train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                          device, pad_idx, args=args, epoch=epoch)
                else:
                    train_meters = single_epoch_debug(model, data_loaders['train'], criteria, optimizer,
                                                      device, pad_idx, args=args, epoch=epoch)

                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']

                if not args.warmup:
                    # lr_scheduler.step(epoch=epoch, metrics=eval_acc)
                    lr_scheduler.step(epoch=epoch)  # Multi-step LR scheduler no metrics

                # lr_scheduler.step(eval_acc)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    # Save the model (overwrite the best one)
                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                     best_test_acc=best_test_acc)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)
                train_meters.update(
                    {'rest_lr': optimizer.param_groups[0]['lr'], 'backbone_lr': optimizer.param_groups[1]['lr']})
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                      main_tag='acc')
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                      step=epoch, main_tag='loss')
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_lr' in k},
                                      step=epoch, main_tag='lr')

                bar.refresh()

                if no_improvement == args.patience:
                    logger.warning(colored('Stopping the training @epoch-{} due to lack of progress in test-accuracy '
                                           'boost (patience hit {} epochs)'.format(epoch, args.patience),
                                           'red', attrs=['bold', 'underline']))
                    break

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            msg = ('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch))
            f_out.write(msg)

        logger.info('Finished training successfully. Good job!')

    elif args.mode == 'evaluate':

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
        logger.info('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        logger.info('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        logger.info('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))

        # out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        if args.analyze_evaluate:
            out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
            res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                      args, out_file=out_file)
            logger.info(res)
