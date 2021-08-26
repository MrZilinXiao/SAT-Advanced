import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from argparse import ArgumentParser
from utils import str2bool, create_dir
from loguru import logger
from termcolor import colored


def parse_arguments(notebook_options=None):
    """Parse the arguments for the training (or test) execution of a ReferIt3D net.
    :param notebook_options: (list) e.g., ['--max-distractors', '100'] to give/parse arguments from inside a jupyter notebook.
    :return:
    """
    parser = argparse.ArgumentParser(description='ReferIt3D Nets + Ablations')

    #
    # self-defined params for exp ease
    #

    parser.add_argument('--exp-name', type=str, help='Experiment name', default='default')
    parser.add_argument('--note', type=str, help='Experiment note', default='default_note')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Using torch.autograd.profiler.profile to profile performance')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--skip-train', action='store_true', default=False)  # do not training, just eval

    #
    # CLIP related options
    #
    parser.add_argument('--clip-backbone', type=str, default=None, choices=["RN50", "RN101", "RN50x4",
                                                                            "RN50x16", "ViT-B/32",
                                                                            "ViT-B/16"])  # if not None, ready to deploy CLIP
    # following are available when clip_backbone is not None
    parser.add_argument('--use-clip-visual', action='store_true', default=False)  # online 2D CLIP feature
    parser.add_argument('--use-clip-language', action='store_true',
                        default=False)  # replace TextBERT, we now only possess `RN50x16` offline feature
    parser.add_argument('--direct-eos', action='store_true', default=False)   # only applicable if use-clip-language
    parser.add_argument('--init-language', action='store_true', default=False)
    parser.add_argument('--rgb-path', type=str, default="")   # if online feature, we need 2D RGB input

    #
    # Non-optional arguments
    #
    parser.add_argument('-scannet-file', type=str, required=True, help='pkl file containing the data of Scannet'
                                                                       ' as generated by running XXX')
    parser.add_argument('-referit3D-file', type=str, required=True)
    parser.add_argument('-offline-2d-feat', type=str, default=None,
                        required=True)  # temporary solution for offline 2D feat
    parser.add_argument('-offline-cache', action='store_true',
                        default=False)  # if caching in /dev/shm, will override offline-2d-feat

    ## TODO: online CLIP feature distillation

    #
    # I/O file-related arguments
    #
    parser.add_argument('--log-dir', type=str, help='where to save training-progress, model, etc')
    parser.add_argument('--resume-path', type=str, help='model-path to resume')
    parser.add_argument('--config-file', type=str, default=None, help='config file')
    parser.add_argument('--pretrain-path', type=str, help='model-path to pretrain')

    #
    # Dataset-oriented arguments
    #

    parser.add_argument('--max-distractors', type=int, default=51,
                        help='Maximum number of distracting objects to be drawn from a scan.')
    parser.add_argument('--max-seq-len', type=int, default=24,
                        help='utterances with more tokens than this they will be ignored.')
    parser.add_argument('--points-per-object', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')
    parser.add_argument('--unit-sphere-norm', type=str2bool, default=False,
                        help="Normalize each point-cloud to be in a unit sphere.")
    parser.add_argument('--mentions-target-class-only', type=str2bool, default=True,
                        help='If True, drop references that do not explicitly mention the target-class.')
    parser.add_argument('--min-word-freq', type=int, default=3)
    parser.add_argument('--max-test-objects', type=int, default=88)

    #
    # Training arguments
    #

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--max-train-epochs', type=int, default=100, help='number of training epochs. [default: 100]')
    parser.add_argument('--n-workers', type=int, default=-1,
                        help='number of data loading workers [default: -1 is all cores available -1.]')
    parser.add_argument('--random-seed', type=int, default=13,
                        help='Control pseudo-randomness (net-wise, point-cloud sampling etc.) fostering reproducibility.')
    parser.add_argument('--init-lr', type=float, default=0.0001, help='learning rate for training.')
    parser.add_argument('--patience', type=int, default=100,
                        help='if test-acc does not improve for patience consecutive'
                             'epoch, stop training. fixed step if patience==max epoch')
    parser.add_argument("--warmup", action="store_true", default=False, help="if lr linear warmup.")

    #
    # Model arguments
    #
    parser.add_argument('--model', type=str, default='referIt3DNet', choices=['referIt3DNet',
                                                                              'directObj2Lang',
                                                                              'referIt3DNetAttentive',
                                                                              'mmt_referIt3DNet'])
    parser.add_argument('--object-latent-dim', type=int, default=768)
    parser.add_argument('--language-latent-dim', type=int, default=768)
    parser.add_argument('--mmt-latent-dim', type=int, default=768)
    # parser.add_argument('--word-embedding-dim', type=int, default=64)
    # parser.add_argument('--graph-out-dim', type=int, default=128)
    # parser.add_argument('--dgcnn-intermediate-feat-dim', nargs='+', type=int, default=[128, 128, 128, 128])

    parser.add_argument('--object-encoder', type=str, default='pnet_pp', choices=['pnet_pp', 'pnet'])
    # parser.add_argument('--language-fusion', type=str, default='both', choices=['before', 'after', 'both'])
    # parser.add_argument('--word-dropout', type=float, default=0.1)
    # parser.add_argument('--knn', type=int, default=7, help='For DGCNN number of neighbors')
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')
    parser.add_argument("--transformer", action="store_true", default=False, help="transformer mmt fusion module.")
    parser.add_argument("--pretrain", action="store_true", default=False, help="if pretrain with MLM (RPP TODO).")
    parser.add_argument('--context_obj', type=str, default=None, help="context object; rand, closest, farthest.")
    # parser.add_argument('--feat2d', type=str, default="ROI", choices=['ROI', 'clsvec', 'clsvecROI', 'ROI3D', 'clsvec3D', 'clsvecROI3D'], help="ROI/clsvec/clsvecROI/ROI3D/clsvec3D/clsvecROI3D. \
    #     XXX3D should be used in unaligned setting as supervision, instead of aligned setting as input")
    parser.add_argument('--feat2d', type=str, default=None,
                        choices=['ROI', 'ROI3D', 'CLIP', 'CLIP3D', 'CLIP_add', 'CLIP_add3D'], help="\
        XXX3D should be used in unaligned setting as supervision, instead of aligned setting as input")  # 新的settings只留可能用到的部分
    parser.add_argument('--clsvec2d', action="store_true", default=True,
                        help='whether append one-hot class vector in 2D feature')

    parser.add_argument('--norm-offline-feat', action="store_true", default=False)

    parser.add_argument('--context_2d', type=str, default=None,
                        help="how to use 2D context; None, aligned or unaligned.")
    parser.add_argument('--mmt_mask', type=str, default=None, help="if apply certain mmt mask.")
    parser.add_argument("--tokenvisualloss", action="store_true", default=False,
                        help="if add paired text token visual region similairty.")
    parser.add_argument("--loss_proj", action="store_true", default=False,
                        help="if add learnable projection matrix in loss.")
    parser.add_argument("--addlabel_words", action="store_true", default=False,
                        help="if add the label words input (from 3D proposals).")

    #
    # Misc arguments
    #

    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device. [default: 0]')
    parser.add_argument('--n-gpus', type=int, default=1, help='number gpu devices. [default: 1]')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per gpu. [default: 32]')
    # parser.add_argument('--save-args', type=str2bool, default=True, help='save arguments in a json.txt')
    parser.add_argument('--save-args', action='store_true', default=False, help='save arguments in a json.txt')
    parser.add_argument('--experiment-tag', type=str, default=None, help='will be used to name a subdir '
                                                                         'for log-dir if given')
    parser.add_argument('--wandb-log', action='store_true', default=False)
    parser.add_argument('--git-commit', action='store_true', default=False)
    parser.add_argument('--analyze-evaluate', action='store_true', default=False)
    # parser.add_argument('--wandb-')
    # parser.add_argument('--cluster-pid', type=str, default=None)

    #
    # "Joint" (Sr3d+Nr3D) training
    #
    parser.add_argument('--augment-with-sr3d', type=str, default=None,
                        help='csv with sr3d data to augment training data'
                             'of args.referit3D-file')
    parser.add_argument('--vocab-file', type=str, default=None, help='optional, .pkl file for vocabulary (useful when '
                                                                     'working with multiple dataset and single model.')
    parser.add_argument('--fine-tune', type=str2bool, default=False,
                        help='use if you train with dataset x and then you '
                             'continue training with another dataset')
    parser.add_argument('--s-vs-n-weight', type=float, default=None, help='importance weight of sr3d vs nr3d '
                                                                          'examples [use less than 1]')

    # Parse args
    if notebook_options is not None:
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()

    if args.mode == 'train' and not args.resume_path and not args.log_dir:
        raise ValueError('You have to indicate either resume_path or log_dir when training!')

    if args.config_file is not None:
        with open(args.config_file, 'r') as fin:
            configs_dict = json.load(fin)
            apply_configs(args, configs_dict)

    # Create logging related folders and arguments
    if args.log_dir:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        if args.pretrain:
            args.log_dir = args.log_dir
        elif args.experiment_tag:
            args.log_dir = osp.join(args.log_dir, args.experiment_tag, timestamp)
        else:
            args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'))

    if args.resume_path and not args.log_dir:  # resume and continue training in previous log-dir.
        checkpoint_dir = osp.split(args.resume_path)[0]  # '/xxx/yyy/log_dir/checkpoints/model.pth'
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = osp.split(checkpoint_dir)[0]
        args.tensorboard_dir = osp.join(args.log_dir, 'tb_logs')

    if args.use_clip_visual or args.use_clip_language:
        if args.clip_backbone is None:
            raise ValueError('You can not use clip online features if not indicating clip_backbone!')

    if args.use_clip_visual:
        if args.rgb_path == "":
            raise ValueError("You need to indicate rgb_path when use_clip_visual is enabled!")
        logger.info("Using {} as RGB input...".format(args.rgb_path))

    if not args.use_clip_language and args.direct_eos:
        raise ValueError("direct-eos is only applicable if use-clip-language is enabled!")

    if args.clip_backbone is not None:
        if not (args.use_clip_visual or args.use_clip_language):  # indicate a backbone but not using it
            logger.warning(
                colored('You indicate a CLIP online backbone {} but not using it!'.format(args.clip_backbone), 'red'))

        if args.use_clip_language:
            args.max_seq_len = 77
            logger.warning(
                colored('You are overriding max_seq_len since you use CLIP language encoder!', 'red'))

        if args.use_clip_visual and args.feat2d is not None:
            logger.warning(
                colored('You enabled CLIP online visual encoder which will OVERRIDE your `feat2d` setting!', 'red'))

    if args.offline_2d_feat is not None and args.offline_cache:
        logger.warning(colored('Offline Memory Caching Enabled...Overriding `offline_2d_feat`...', 'red'))

    # Print them nicely
    args_string = pprint.pformat(vars(args))
    print(args_string)

    if args.save_args:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args


def read_saved_args(config_file, override_args=None, verbose=True):
    """
    :param config_file:
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)
