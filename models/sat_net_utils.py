"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import pandas as pd
import tqdm
import math
import sys

current_module = sys.modules[__name__]
import torch.nn.functional as F

from utils.evaluation import AverageMeter
from loguru import logger


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    # batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    batch_keys = ['objects', 'tokens', 'target_pos', 'token_inds', 'token_num', 'context_size',
                  'obj_offset']  # all models use these
    batch_keys.extend(['context_objects', 'closest_context_objects', 'farthest_context_objects', 'rand_context_objects', \
                       'context_offset', 'closest_context_offset', 'farthest_context_offset', 'rand_context_offset', \
                       'feat_2d', 'coords_2d', 'word_split_inds'])
    if args.pretrain:
        batch_keys.append('mlm_label')
        batch_keys.append('contra_pollute')
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, args, epoch=None):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    # np.random.seed()  # call this to change the sampling of the point-clouds
    np.random.seed(
        args.random_seed + epoch)  # call this to change the sampling of the point-clouds, in a time-invariant way
    batch_keys = make_batch_keys(args)
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:  # is this indicating a performance issue?
    for batch in tqdm.tqdm(data_loader):
        # for batch in data_loader:
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        # Backward
        optimizer.zero_grad()
        all_losses = compute_losses(batch, res, criteria, args)
        total_loss = all_losses['total_loss']
        total_loss.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(total_loss.item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        # TODO copy the ref-loss to homogeneize the code
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        if args.pretrain:  ## MLM Acc
            batch_size, tokenlen, vocab_size = res['mlm_pred'].shape
            scores = res['mlm_pred'].view(batch_size * tokenlen, -1)
            targets = batch['mlm_label'].view(batch_size * tokenlen)
            loss_mask = (targets != -1).float()
            pollute_mask = batch["contra_pollute"].repeat(1, tokenlen).view(-1).float()
            loss_mask = (loss_mask * (1 - pollute_mask)).bool()  ## 1 is polluted, not compute MLM
            if torch.sum(loss_mask.float()) == 0.:
                accuracy = torch.zeros(1, device=torch.device('cuda'))
            else:
                scores, targets = scores[loss_mask], targets[loss_mask]
                accuracy = torch.sum(scores.argmax(1) == targets).float() / max(targets.shape[0], 1)
            guessed_correctly = accuracy.item()  # .cpu().numpy()
            ## contra acc
            contra_accuracy = torch.sum(((torch.sigmoid(res['contra_pred']) > 0.5).float() == (
                    batch['contra_pollute'] > 0.5).float()).float()) / max(res['contra_pred'].shape[0], 1)
            # ref_acc_mtr.update(guessed_correctly, batch_size)
            ref_acc_mtr.update(contra_accuracy * accuracy.item(), batch_size)
        else:  ## main ref acc
            predictions = torch.argmax(res['logits'], dim=1)
            guessed_correctly = torch.mean((predictions == target).double()).item()
            ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_loss'] = referential_loss_mtr.avg
    metrics['train_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return metrics


# ## Learning Transferable Visual Models From Natural Language Supervision
# ## Figure 3, without learned projection
# def clip_matching_loss(feat1, feat2, obj_count, temperature=0.07):
#     sim_losses = 0.
#     feat1 = F.normalize(feat1, p=2, dim=-1)
#     feat2 = F.normalize(feat2, p=2, dim=-1)
#     for b_i in range(feat1.shape[0]):
#         feat_2d, feat_3d, num_obj = feat1[b_i, :, :], feat2[b_i, :, :], obj_count[b_i]
#         feat_2d, feat_3d = feat_2d[:num_obj, :], feat_3d[:num_obj, :]
#         ################################
#         ## own implementation of CE
#         ## // if as in 480-eq3, difference is that this version CE over each positive, while NTXent sum over 64 positive as numerator
#         ## TODO: so should be same with 1 positive sample, to verify
#         ################################
#         # logits = feat_2d.mm(feat_3d.t()) * math.exp(temperature)
#         # labels = torch.arange(num_obj, device=torch.device('cuda'))
#         # loss_i = F.cross_entropy(logits.unsqueeze(0),labels.unsqueeze(0))
#         # loss_t = F.cross_entropy(logits.t().unsqueeze(0),labels.unsqueeze(0))
#         # sim_losses = sim_losses + (loss_i + loss_t)/2
#         ## NT-Xent, expect to be same but ~3X larger...
#         loss_func = losses.NTXentLoss(temperature=temperature)
#         embeddings = torch.cat((feat_2d, feat_3d))
#         indices = torch.arange(0, num_obj, device=torch.device('cuda'))
#         labels = torch.cat((indices, indices))
#         loss = loss_func(embeddings, labels)
#         sim_losses = sim_losses + loss
#     return 0.5 * sim_losses / (feat1.shape[0])
# ## partial batched version
# sim_losses = 0.
# feat1 = F.normalize(feat1, p=2, dim=-1)
# feat2 = F.normalize(feat2, p=2, dim=-1)
# per_loss_sample = 2
# for b_i in range(feat1.shape[0]//per_loss_sample):
#     feat_2d_list, feat_3d_list = [], []
#     for ii in range(per_loss_sample):
#         sample_idx = b_i*per_loss_sample+ii
#         feat_2d, feat_3d, num_obj = feat1[sample_idx,:,:], feat2[sample_idx,:,:], obj_count[sample_idx]
#         feat_2d_list.append(feat_2d[:num_obj,:])
#         feat_3d_list.append(feat_3d[:num_obj,:])
#     feat_2d = torch.cat(feat_2d_list, dim=0)
#     feat_3d = torch.cat(feat_3d_list, dim=0)
#     loss_func = losses.NTXentLoss(temperature=temperature)
#     embeddings = torch.cat((feat_2d, feat_3d))
#     indices = torch.arange(0, feat_2d.shape[0], device=torch.device('cuda'))
#     labels = torch.cat((indices, indices))
#     loss = loss_func(embeddings, labels)
#     sim_losses = sim_losses + loss
# return 0.5*sim_losses/(feat1.shape[0]//per_loss_sample)


def contrastive_loss(feat1, feat2, obj_count, margin=0.1, max_margin=True, weight=10., reduction=True):
    """
    2D<->3D corr loss，一个triplet loss
    reduction返回single value
    """
    sim_losses = 0. if reduction else []
    ## should norm? to be tested// L2 norm === 27.71 (sqrt(768)); w/o norm like almost no margin
    ## after adding norm, adjust weight accordingly (margin *6 from directly loss scale)

    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)
    for b_i in range(feat1.shape[0]):
        feat_2d, feat_3d, num_obj = feat1[b_i, :, :], feat2[b_i, :, :], obj_count[b_i]
        feat_2d, feat_3d = feat_2d[:num_obj, :], feat_3d[:num_obj, :]
        cos_scores = feat_2d.mm(feat_3d.t())
        diagonal = cos_scores.diag().view(feat_2d.size(0), 1)
        d1 = diagonal.expand_as(cos_scores)
        d2 = diagonal.t().expand_as(cos_scores)
        # feat_3d retrieval
        cost_3d = (margin + cos_scores - d1).clamp(min=0)
        # feat2d retrieval
        cost_2d = (margin + cos_scores - d2).clamp(min=0)
        # clear diagonals
        I = (torch.eye(cos_scores.size(0), device=torch.device('cuda')) > .5)
        cost_3d = cost_3d.masked_fill_(I, 0)
        cost_2d = cost_2d.masked_fill_(I, 0)
        # keep the maximum violating negative for each query
        # consider both in sample, and cross-sample 2d-3d "retrival"
        if False:
            cost_3d = cost_3d.max(1)[0]
            cost_2d = cost_2d.max(0)[0]
        else:
            topk = min(3, int(cost_3d.shape[0]))
            cost_3d = (torch.topk(cost_3d, topk, dim=1)[0])
            cost_2d = (torch.topk(cost_2d, topk, dim=0)[0])
        if reduction:
            batch_loss = torch.sum(cost_3d) + torch.sum(cost_2d)
            sim_losses = sim_losses + batch_loss
        else:
            batch_loss = torch.mean(cost_3d) + torch.mean(cost_2d)
            sim_losses.append(batch_loss)
    if reduction:
        return weight * sim_losses / (torch.sum(obj_count))
    else:
        return weight * torch.tensor(sim_losses, device=torch.device('cuda'))


def contrastive_loss_batch(feat1, feat2, obj_count, margin=0.1, max_margin=True, weight=10., reduction=True):
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)
    feat_2d_list, feat_3d_list = [], []
    for b_i in range(feat1.shape[0]):
        feat_2d, feat_3d, num_obj = feat1[b_i, :, :], feat2[b_i, :, :], obj_count[b_i]
        feat_2d_list.append(feat_2d[:num_obj, :])
        feat_3d_list.append(feat_3d[:num_obj, :])
    feat_2d = torch.cat(feat_2d_list, dim=0)
    feat_3d = torch.cat(feat_3d_list, dim=0)
    cos_scores = feat_2d.mm(feat_3d.t())
    diagonal = cos_scores.diag().view(feat_2d.size(0), 1)
    d1 = diagonal.expand_as(cos_scores)
    d2 = diagonal.t().expand_as(cos_scores)
    # feat_3d retrieval
    cost_3d = (margin + cos_scores - d1).clamp(min=0)
    # feat2d retrieval
    cost_2d = (margin + cos_scores - d2).clamp(min=0)
    # clear diagonals
    I = (torch.eye(cos_scores.size(0), device=torch.device('cuda')) > .5)
    cost_3d = cost_3d.masked_fill_(I, 0)
    cost_2d = cost_2d.masked_fill_(I, 0)
    # keep the maximum violating negative for each query
    # consider both in sample, and cross-sample 2d-3d "retrival"
    # if True:
    #     cost_3d = cost_3d.max(1)[0]
    #     cost_2d = cost_2d.max(0)[0]
    # else:
    #     cost_3d = (torch.topk(cost_3d, 10, dim=1)[0])/10.
    #     cost_2d = (torch.topk(cost_2d, 10, dim=0)[0])/10.
    # return weight * (torch.sum(cost_3d) + torch.sum(cost_2d)) / (torch.sum(obj_count))
    if True:
        cost_3d = cost_3d.max(1)[0]
        cost_2d = cost_2d.max(0)[0]
    else:
        cost_3d = (torch.topk(cost_3d, 3, dim=1)[0])
        cost_2d = (torch.topk(cost_2d, 3, dim=0)[0])
    if reduction:
        return weight * (torch.sum(cost_3d) + torch.sum(cost_2d)) / (torch.sum(obj_count))
    else:
        raise NotImplementedError("Not implemented when cost3d is not top-1")
        return weight * (torch.mean(cost_3d, dim=1) + torch.mean(cost_2d, dim=1))


def merge_wordsplit_feat(token_feat, word_split_inds, context_size, proj_token_feat=None):
    # print('WARNING: do not follow strict mean word split feature for speed; do not use in benchmark exps!!!\n add_special_tokens=False)[0]] in loader, merge_wordsplit_feat\n')
    return token_feat, proj_token_feat, context_size
    truncate_context_size = torch.zeros(context_size.shape, device=torch.device('cuda')).long()
    return_token_feat = torch.zeros(token_feat.shape, device=torch.device('cuda'))
    if proj_token_feat is not None:
        proj_return_token_feat = torch.zeros(proj_token_feat.shape, device=torch.device('cuda'))
    for b_i in range(token_feat.shape[0]):
        truncate_context_size[b_i] = min(context_size[b_i], word_split_inds[b_i, :].max() + 1)
        for c_i in range(word_split_inds[b_i, :].max() + 1):
            idx = (word_split_inds[b_i, :] == c_i)
            return_token_feat[b_i, c_i, :] = torch.mean(token_feat[b_i, idx, :], dim=0)
            if proj_token_feat is not None:
                proj_return_token_feat[b_i, c_i, :] = torch.mean(proj_token_feat[b_i, idx, :], dim=0)
    if proj_token_feat is not None:
        return return_token_feat, proj_return_token_feat, truncate_context_size
    else:
        return return_token_feat, truncate_context_size


def compute_losses(batch, res, criterion_dict, args):
    """Calculate the loss given the model logits and the criterion
    :param batch:
    :param res: dict of logits
    :param criterion_dict: dict of the criterion should have key names same as the logits
    :param args, argparse.Namespace
    :return: scalar loss value
    """
    # Get the object language classification loss and the object classification loss
    criterion = criterion_dict['logits']
    logits = res['logits']

    # Panos-note investigating tb output (if you do it like this, it does not separate, later additions
    # to total_loss from TODO POST DEADLINE.
    # referential_loss)
    # referential_loss = criterion(logits, batch['target_pos'])
    # total_loss = referential_loss
    if args.pretrain:  # 用MLM预训练？
        scores = res['mlm_pred'].permute(0, 2, 1)
        targets = batch['mlm_label']
        loss_mask = (targets != -1).float()
        pollute_mask = batch["contra_pollute"].repeat(1, loss_mask.shape[-1]).float()
        loss_mask = (loss_mask * (1 - pollute_mask))  ## 1 is polluted, not compute MLM
        losses = F.cross_entropy(
            scores, targets, reduction="none", ignore_index=-1
        )  # ignore_index=-1
        losses *= loss_mask  # .unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), torch.Tensor([1.]).to(losses.device))
        ## borrow ref_loss for visu perpose
        total_loss = torch.sum(losses) / count
        ## contra loss
        contra_loss = F.binary_cross_entropy_with_logits(
            res['contra_pred'], batch['contra_pollute'].float(), reduction="mean")
        total_loss += contra_loss
    else:
        ## if apply mask on padded regions
        mainloss_mask = None
        if False:
            mainloss_mask = (batch['class_labels'] != res['class_logits'].shape[-1]).bool()
        if args.s_vs_n_weight is not None:
            # total_loss = criterion(logits, batch)
            total_loss = criterion_dict['logits_nondec'](logits, batch['target_pos'])
        else:
            total_loss = criterion(logits, batch['target_pos'])

    ## CE/FL, pad position shouldn't be calculated
    # sim_loss_type = 'fixed_ce'
    sim_loss_type = 'contrastive'

    if sim_loss_type == 'contrastive':
        simloss = getattr(current_module, 'contrastive_loss')  # 定义simloss
        # simloss = getattr(current_module, 'contrastive_loss_batch')
    elif sim_loss_type == 'fixed_ce':
        simloss = getattr(current_module, 'clip_matching_loss')

    if args.context_2d == 'unaligned':  # 默认没有对齐
        ## 2D-lang align loss
        # vg2d_loss = criterion(res['logits_2D'], batch['target_pos'])
        if args.s_vs_n_weight is not None:
            # vg2d_loss = criterion(res['logits_2D'], batch)
            vg2d_loss = criterion_dict['logits_nondec'](logits, batch[
                'target_pos'])  # nn.CrossEntropyLoss(reduction='none').to(device)
        else:
            vg2d_loss = criterion(res['logits_2D'], batch['target_pos'])
            # 2D Grounding loss, logits_2D(softmax后)对齐target_pos(语句的GT object choice)
            # criterion默认是‘class_logits’
        ## 2D-3D align loss; contra loss, cos
        feat_2d = res['mmt_obj_output_2D']  # 虽然是offline的2D feature，但是后面接了一个head
        feat_3d = res['mmt_obj_output']
        feat_texttoken_2d = res['mmt_texttoken2d_output']
        feat_texttoken_3d = res['mmt_texttoken3d_output']
        # feat_texttoken_2d, feat_texttoken_3d = res['mmt_texttoken_output'], res['mmt_texttoken_output']

        ## if too many word split and tokens are truncated, batch_context_size will be smaller than batch['context_size']
        # feat_texttoken, batch_context_size = merge_wordsplit_feat(\
        #     res['mmt_texttoken_output'], batch['word_split_inds'], batch['context_size'])
        sim2d3d_loss = simloss(feat_2d, feat_3d, batch['context_size'], reduction=(args.s_vs_n_weight is None))

        # total_loss = total_loss + vg2d_loss
        # print(total_loss)
        total_loss = total_loss + vg2d_loss + sim2d3d_loss  # * 0.1
        if args.tokenvisualloss:  # default False
            feat_texttoken_2d, feat_texttoken_3d, batch_context_size = merge_wordsplit_feat( \
                feat_texttoken_2d, batch['word_split_inds'], batch['context_size'], proj_token_feat=feat_texttoken_3d)
            total_loss += simloss(feat_2d, feat_texttoken_2d, batch_context_size,
                                  reduction=(args.s_vs_n_weight is None))  # * 0.1
            total_loss += simloss(feat_3d, feat_texttoken_3d, batch_context_size,
                                  reduction=(args.s_vs_n_weight is None))  # * 0.1

    elif args.context_2d == 'aligned':
        if args.tokenvisualloss:
            # feat_texttoken, batch_context_size = merge_wordsplit_feat(res['mmt_texttoken_output'], batch['word_split_inds'], batch['context_size'])
            # total_loss += simloss(res['mmt_obj_output'], feat_texttoken, batch_context_size) * 0.1
            feat_texttoken_2d, feat_texttoken_3d, batch_context_size = merge_wordsplit_feat( \
                res['mmt_texttoken2d_output'], batch['word_split_inds'], batch['context_size'],
                proj_token_feat=res['mmt_texttoken3d_output'])
            total_loss += simloss(res['mmt_obj_output'], feat_texttoken_3d, batch_context_size,
                                  reduction=(args.s_vs_n_weight is None))
    else:
        if args.tokenvisualloss:
            # feat_texttoken, batch_context_size = merge_wordsplit_feat(res['mmt_texttoken_output'], batch['word_split_inds'], batch['context_size'])
            # total_loss += simloss(res['mmt_obj_output'], feat_texttoken, batch_context_size) * 0.1
            feat_texttoken_2d, feat_texttoken_3d, batch_context_size = merge_wordsplit_feat( \
                res['mmt_texttoken2d_output'], batch['word_split_inds'], batch['context_size'],
                proj_token_feat=res['mmt_texttoken3d_output'])
            total_loss += simloss(res['mmt_obj_output'], feat_texttoken_3d, batch_context_size,
                                  reduction=(args.s_vs_n_weight is None))
    if args.s_vs_n_weight is not None:
        weights = torch.ones(total_loss.shape).to(total_loss.device)
        weights[batch['is_nr3d']] = 1. / args.s_vs_n_weight
        total_loss = total_loss * weights
        total_loss = total_loss.sum() / len(total_loss)

    referential_loss = total_loss.item()
    obj_clf_loss = lang_clf_loss = 0

    # 原先ReferIt3D的loss
    if args.obj_cls_alpha > 0:
        criterion = criterion_dict['class_logits']
        obj_clf_loss = criterion(res['class_logits'].transpose(2, 1), batch['class_labels'])
        if args.s_vs_n_weight is not None:
            obj_clf_loss = torch.mean(obj_clf_loss, dim=1) * weights
            obj_clf_loss = obj_clf_loss.sum() / len(obj_clf_loss)
        total_loss += obj_clf_loss * args.obj_cls_alpha

    if args.lang_cls_alpha > 0:
        criterion = criterion_dict['lang_logits']
        lang_clf_loss = criterion(res['lang_logits'], batch['target_class'])
        if args.s_vs_n_weight is not None:
            lang_clf_loss = lang_clf_loss * weights
            lang_clf_loss = lang_clf_loss.sum() / len(lang_clf_loss)
        total_loss += lang_clf_loss * args.lang_cls_alpha

    return {'total_loss': total_loss, 'referential_loss': referential_loss,
            'obj_clf_loss': obj_clf_loss, 'lang_clf_loss': lang_clf_loss}


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.eval()

    assert (randomize == False)
    if randomize:
        np.random.seed()  # call this to change the sampling of the point-clouds #TODO-A talk about it.
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        all_losses = compute_losses(batch, res, criteria, args)

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(all_losses['total_loss'].item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        if args.pretrain:  ## MLM Acc
            batch_size, tokenlen, vocab_size = res['mlm_pred'].shape
            scores = res['mlm_pred'].view(batch_size * tokenlen, -1)
            targets = batch['mlm_label'].view(batch_size * tokenlen)
            loss_mask = (targets != -1).float()
            pollute_mask = batch["contra_pollute"].repeat(1, tokenlen).view(-1).float()
            loss_mask = (loss_mask * (1 - pollute_mask)).bool()  ## 1 is polluted, not compute MLM
            if torch.sum(loss_mask.float()) == 0.:
                accuracy = torch.zeros(1, device=torch.device('cuda'))
            else:
                scores, targets = scores[loss_mask], targets[loss_mask]
                accuracy = torch.sum(scores.argmax(1) == targets).float() / max(targets.shape[0], 1)
            guessed_correctly = accuracy.item()  # .cpu().numpy()
            ## contra acc
            contra_accuracy = torch.sum(((torch.sigmoid(res['contra_pred']) > 0.5).float() == (
                    batch['contra_pollute'] > 0.5).float()).float()) / max(res['contra_pred'].shape[0], 1)
            # ref_acc_mtr.update(guessed_correctly, batch_size)
            ref_acc_mtr.update(contra_accuracy * accuracy.item(), batch_size)
        else:  ## main ref acc
            predictions = torch.argmax(res['logits'], dim=1)
            guessed_correctly = torch.mean((predictions == target).double()).item()
            ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_loss'] = referential_loss_mtr.avg
    metrics['test_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        out = model(batch)

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        #     ## TODO: add a txt write here with format, scene-id, object-id, query, correctness
        #     pred_txt = open('tmp.txt','a')
        #     # pred_txt = open('pred_contra10_475.txt','w')
        #     # print(predictions)
        #     # print(batch['target_pos'])
        #     guessed_correctly = (predictions == batch['target_pos']).long().cpu().numpy()
        #     for ii in range(len(batch['utterance'])):
        #         # print(batch['stimulus_id'][ii],batch['object_ids'][ii],len(batch['object_ids'][ii]))
        #         # print(int(predictions[ii]),int(batch['target_pos'][ii]))
        #         # print(batch['stimulus_id'][ii])
        #         # print(batch['object_ids'][ii])
        #         # print(batch['target_object_id'][ii])
        #         # print(batch['object_ids'][ii][int(batch['target_pos'][ii])])
        #         # exit(0)
        #         query = batch['utterance'][ii]
        #         sampleid = batch['stimulus_id'][ii]
        #         correct = int(guessed_correctly[ii])
        #         # predid,gtid = int(predictions[ii]), int(batch['target_pos'][ii])
        #         predid,gtid = int(batch['object_ids'][ii][int(predictions[ii])]), int(batch['object_ids'][ii][int(batch['target_pos'][ii])])
        #         # pred_txt.write('%s,%s,%s\n'%(sampleid,correct,query))
        #         pred_txt.write('%s,%d,%d,%d,%s\n'%(sampleid,correct,predid,gtid,query))
        #     pred_txt.close()

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    # exit(0)

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])

    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=13):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
