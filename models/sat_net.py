import torch
import random
import argparse
from torch import nn
from collections import defaultdict

# from pytorch_transformers.modeling_bert import (
#     BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
#     BertPreTrainedModel
# )

from models.mmt_module import *

# from . import DGCNN
from models.default_blocks import *
from models.utils import get_siamese_features
from in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


# class ReferIt3DNet(nn.Module):
#     """
#     A neural listener for segmented 3D scans based on graph-convolutions.
#     """
#
#     def __init__(self,
#                  args,
#                  object_encoder,
#                  language_encoder,
#                  graph_encoder,
#                  object_language_clf,
#                  object_clf=None,
#                  language_clf=None):
#         """
#         Parameters have same meaning as in Base3DListener.
#
#         @param args: the parsed arguments
#         @param object_encoder: encoder for each segmented object ([point-cloud, color]) of a scan
#         @param language_encoder: encoder for the referential utterance
#         @param graph_encoder: the graph net encoder (DGCNN is the used graph encoder)
#         given geometry is the referred one (typically this is an MLP).
#         @param object_clf: classifies the object class of the segmented (raw) object (e.g., is it a chair? or a bed?)
#         @param language_clf: classifies the target-class type referred in an utterance.
#         @param object_language_clf: given a fused feature of language and geometry, captures how likely it is that the
#         """
#
#         super().__init__()
#
#         # The language fusion method (either before the graph encoder, after, or in both ways)
#         self.language_fusion = args.language_fusion
#
#         # Encoders
#         self.object_encoder = object_encoder
#         self.language_encoder = language_encoder
#         self.graph_encoder = graph_encoder
#
#         # Classifier heads
#         self.object_clf = object_clf
#         self.language_clf = language_clf
#         self.object_language_clf = object_language_clf
#
#     def __call__(self, batch: dict) -> dict:
#         result = defaultdict(lambda: None)
#
#         # Get features for each segmented scan object based on color and point-cloud
#         objects_features = get_siamese_features(self.object_encoder, batch['objects'],
#                                                 aggregator=torch.stack)  # B X N_Objects x object-latent-dim
#
#         # Classify the segmented objects
#         if self.object_clf is not None:
#             objects_classifier_features = objects_features
#             result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)
#
#         # Get feature for utterance
#         n_objects = batch['objects'].size(1)
#         lang_features = self.language_encoder(batch['tokens'])
#         lang_features_expanded = torch.unsqueeze(lang_features, -1).expand(-1, -1, n_objects).transpose(
#             2, 1)  # B X N_Objects x lang-latent-dim
#
#         # Classify the target instance label based on the text
#         if self.language_clf is not None:
#             result['lang_logits'] = self.language_clf(lang_features)
#
#         # Start graph encoding
#         graph_visual_in_features = objects_features
#         if self.language_fusion == 'before' or self.language_fusion == 'both':
#             graph_in_features = torch.cat([graph_visual_in_features, lang_features_expanded], dim=-1)
#         else:
#             graph_in_features = graph_visual_in_features
#
#         graph_out_features = self.graph_encoder(graph_in_features)
#
#         if self.language_fusion in ['after', 'both']:
#             final_features = torch.cat([graph_out_features, lang_features_expanded], dim=-1)
#         else:
#             final_features = graph_out_features
#
#         result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)
#         # print(batch['objects'].shape,batch['tokens'].shape,objects_features.shape,lang_features.shape,result['logits'].shape)
#         # torch.Size([64, 88, 1024, 6]) torch.Size([64, 26]) torch.Size([64, 88, 128]) torch.Size([64, 128]) torch.Size([64, 88])
#         return result


class MMT_ReferIt3DNet(nn.Module):  # SAT Model
    """
    A neural listener for segmented 3D scans based on graph-convolutions.
    """

    def __init__(self,
                 args,
                 object_encoder,
                 # language_encoder,
                 # graph_encoder,
                 num_class,
                 object_language_clf=None,
                 object_clf=None,
                 language_clf=None,
                 visudim=128,
                 MMT_HIDDEN_SIZE=192,
                 TEXT_BERT_HIDDEN_SIZE=768,
                 pretrain=False,
                 context_2d=None,
                 feat2dtype=None,
                 mmt_mask=None):
        """
        Parameters have same meaning as in Base3DListener.
        """

        super().__init__()

        # # The language fusion method (either before the graph encoder, after, or in both ways)
        # self.language_fusion = args.language_fusion
        self.args = args
        self.loss_proj = args.loss_proj  # default False
        self.args_mode = args.mode
        self.text_length = args.max_seq_len
        self.addlabel_words = args.addlabel_words
        self.pretrain = pretrain
        self.context_2d = context_2d
        self.feat2dtype = feat2dtype
        self.mmt_mask = mmt_mask
        # Encoders for single object
        self.object_encoder = object_encoder
        self.linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)  #
        self.obj_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_drop = nn.Dropout(0.1)

        # Encoders for visual 2D objects
        # num_class_dim = 525 if '00' in args.scannet_file else 608
        # if (args.feat2d.replace('3D', '')) == 'ROI':
        #     featdim = 2048
        # elif (args.feat2d.replace('3D', '')) == 'clsvec':
        #     featdim = num_class_dim
        # elif (args.feat2d.replace('3D', '')) == 'clsvecROI':
        #     featdim = 2048 + num_class_dim

        num_class_dim = 525 if '00' in args.scannet_file else 608
        print("'00' in args.scannet_file: ", '00' in args.scannet_file)

        if self.args.feat2d.startswith('ROI'):
            featdim = 2048
        elif self.args.feat2d.startswith('CLIP'):  # feat that do not norm!! should consider?
            featdim = 768
        elif self.args.feat2d.startswith('CLIP_add'):
            featdim = 768
        elif self.args.feat2d is None:
            pass
        else:
            raise NotImplemented("Not recognized feat2d keys: {}".format(self.args.feat2d))

        if self.args.clsvec2d:
            featdim += num_class_dim

        self.linear_2d_feat_to_mmt_in = nn.Linear(featdim, MMT_HIDDEN_SIZE)
        # self.linear_2d_feat_to_mmt_in = nn.Sequential(nn.Linear(2048, 32),
        #                          nn.Linear(32, MMT_HIDDEN_SIZE))
        self.linear_2d_bbox_to_mmt_in = nn.Linear(16, MMT_HIDDEN_SIZE)  # 4(bbox) + 12(pose)
        self.obj2d_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj2d_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)

        ## encoder for context object
        self.cnt_object_encoder = single_object_encoder(768)
        self.cnt_linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.cnt_linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
        self.cnt_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.cnt_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.context_drop = nn.Dropout(0.1)

        # Encoders for text
        self.text_bert_config = BertConfig(
            hidden_size=TEXT_BERT_HIDDEN_SIZE,
            num_hidden_layers=3,
            num_attention_heads=12,
            type_vocab_size=2)
        self.text_bert = TextBert.from_pretrained(
            'bert-base-uncased', config=self.text_bert_config, \
            mmt_mask=self.mmt_mask, addlabel_words=self.addlabel_words)
        if TEXT_BERT_HIDDEN_SIZE != MMT_HIDDEN_SIZE:
            self.text_bert_out_linear = nn.Linear(TEXT_BERT_HIDDEN_SIZE, MMT_HIDDEN_SIZE)
        else:
            self.text_bert_out_linear = nn.Identity()

        # ##
        # if args.feat2d=='clsvec':
        #     # print(self.linear_2d_feat_to_mmt_in.weight.shape)
        #     # print(self.text_bert.embeddings.word_embeddings.weight.shape)
        #     idx = list(open('idx_525.txt','r'))
        #     for line in idx:
        #         i,j = int(line.strip().split(',')[0]), int(line.strip().split(',')[1])
        #         self.linear_2d_feat_to_mmt_in.weight.data[:,i] = self.text_bert.embeddings.word_embeddings.weight.data[j,:]

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf
        self.object_language_clf = object_language_clf

        self.mmt_config = BertConfig(
            hidden_size=MMT_HIDDEN_SIZE,
            num_hidden_layers=4,
            num_attention_heads=12,
            type_vocab_size=2)
        self.mmt = MMT(self.mmt_config, context_2d=self.context_2d, mmt_mask=self.mmt_mask,
                       addlabel_words=self.addlabel_words)
        # self.mmt = MMT.from_pretrained(
        #     'bert-base-uncased', config=self.mmt_config
        # )
        # self.lang_cls = nn.Sequential(
        #     nn.Linear(MMT_HIDDEN_SIZE, num_class),
        #     # nn.Dropout()
        # )
        self.matching_cls = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
        if self.context_2d == 'unaligned':
            self.matching_cls_2D = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
        self.mlm_cls = BertLMPredictionHead(self.text_bert.embeddings.word_embeddings.weight,
                                            input_size=MMT_HIDDEN_SIZE)
        # self.mlm_cls = MatchingLinear(outputdim = self.text_bert.embeddings.word_embeddings.weight.shape[0])
        self.contra_cls = PolluteLinear()
        if self.loss_proj:  # 新建一个loss_proj映射？
            # self.fw_2dfeat = nn.Linear(768,768,bias=False)
            # self.fw_3dfeat = nn.Linear(768,768,bias=False)
            if self.addlabel_words:
                self.fw_text2dfeat = nn.Linear(768, 768, bias=False)
                self.fw_text3dfeat = nn.Linear(768, 768, bias=False)
            self.fw_2dfeat = nn.Linear(768, 768)
            self.fw_3dfeat = nn.Linear(768, 768)
            # self.fw_2dfeat = nn.Sequential(nn.Linear(768, 768),
            #                     nn.ReLU(),
            #                     nn.Linear(768, 768))
            # self.fw_3dfeat = nn.Sequential(nn.Linear(768, 768),
            #                     nn.ReLU(),
            #                     nn.Linear(768, 768))

    def __call__(self, batch: dict) -> dict:
        """
        batch带的key解释：
        context_size： samples的数量
        objects：3D object的稀疏采样
        """
        result = defaultdict(lambda: None)

        if self.pretrain:  # don't understand
            batch_size = int(batch['contra_pollute'].shape[0])
            pollute_text_idx = list(range(batch_size))
            pollute_visu_idx = list(range(batch_size))
            for bi in range(batch_size):
                if batch['contra_pollute'][bi] == 1:
                    if random.random() < 0.:
                        pollute_text_idx[bi] = random.choice([i for i in list(range(batch_size)) if i != bi])
                    else:
                        pollute_visu_idx[bi] = random.choice([i for i in list(range(batch_size)) if i != bi])
            batch['objects'] = batch['objects'][pollute_visu_idx]

            batch['token_inds'] = batch['token_inds'][pollute_text_idx]
            batch['token_num'] = batch['token_num'][pollute_text_idx]
            if self.addlabel_words: batch['tag_token_num'] = batch['tag_token_num'][pollute_text_idx]

        # Get features for each segmented scan object based on color and point-cloud: 3D feature
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(objects_features)) + \
                     self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(batch['obj_offset']))
        # 3D features

        if self.context_2d == 'aligned':  # 如果2D-3D已对齐
            obj_mmt_in = obj_mmt_in + \
                         self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                         self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))

        obj_mmt_in = self.obj_drop(obj_mmt_in)
        obj_num = obj_mmt_in.size(1)  # N, obj_num, feat_size
        obj_mask = _get_mask(batch['context_size'].to(obj_mmt_in.device), obj_num)  ## all proposals are non-empty
        # should be all 1, since context_size == obj_num == len(samples)

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = obj_mmt_in
            # objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        if self.context_2d == 'unaligned':  # 将2D feat作为context_obj接到3D feat后面
            context_obj_mmt_in = self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                                 self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))

            if '3D' in self.feat2dtype:
                context_obj_mmt_in += obj_mmt_in  # feat2dtype里面带有3D：在context_2D_feat里面add 3D feat

            context_obj_mmt_in = self.context_drop(context_obj_mmt_in)
            context_obj_mask = _get_mask(batch['context_size'].to(context_obj_mmt_in.device),
                                         obj_num)  ## all proposals are non-empty
            # 均1
            obj_mmt_in = torch.cat([obj_mmt_in, context_obj_mmt_in], dim=1)
            obj_mask = torch.cat([obj_mask, context_obj_mask], dim=1)

        # if 'context_objects' in batch:  # or 'closest_context_objects' in batch or 'rand_context_objects' in batch:
        #     # context objects是一系列3D distractors，包括target_object
        #     # 有必要context objects吗？论文里没提到
        #     context_object = batch[
        #         'context_objects']  # if 'closest_context_objects' in batch else batch['rand_context_objects']
        #     context_objects_features = get_siamese_features(self.cnt_object_encoder, context_object,
        #                                                     aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        #     context_obj_mmt_in = self.cnt_feat_layer_norm(
        #         self.cnt_linear_obj_feat_to_mmt_in(context_objects_features)) + \
        #                          self.cnt_bbox_layer_norm(self.cnt_linear_obj_bbox_to_mmt_in(batch['context_offset']))
        #     context_obj_mmt_in = self.context_drop(context_obj_mmt_in)
        #     context_obj_mask = _get_mask(batch['context_size'].to(context_obj_mmt_in.device),
        #                                  obj_num)  ## all proposals are non-empty
        #     obj_mmt_in = torch.cat([obj_mmt_in, context_obj_mmt_in], dim=1)  # N, obj_num + context_obj_num, feat_size
        #     obj_mask = torch.cat([obj_mask, context_obj_mask], dim=1)

        # Get feature for utterance
        txt_inds = batch["token_inds"]  # batch_size, lang_size
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.

        if not self.addlabel_words:
            txt_mask = _get_mask(batch['token_num'].to(txt_inds.device),
                                 txt_inds.size(1))  ## all proposals are non-empty
        else:
            txt_mask = _get_mask(batch['token_num'].to(txt_inds.device),
                                 txt_inds.size(1) - obj_num)  ## all proposals are non-empty
            tag_txt_mask = _get_mask(batch['tag_token_num'].to(txt_inds.device),
                                     obj_num)  ## all proposals are non-empty
            txt_mask = torch.cat([txt_mask, tag_txt_mask], dim=1)
            txt_type_mask[:, :txt_inds.size(1) - obj_num] = 0
        txt_type_mask = txt_type_mask.long()

        text_bert_out = self.text_bert(
            txt_inds=txt_inds,
            txt_mask=txt_mask,
            txt_type_mask=txt_type_mask
        )
        txt_emb = self.text_bert_out_linear(text_bert_out)
        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(text_bert_out[:, 0, :])

        mmt_results = self.mmt(
            txt_emb=txt_emb,
            txt_mask=txt_mask,
            obj_emb=obj_mmt_in,
            obj_mask=obj_mask,
            obj_num=obj_num
        )

        if self.args_mode == 'evaluate':
            if not self.addlabel_words:
                assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + obj_num))
            else:
                assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + 2 + obj_num * 2))  # why +2 ?
        if self.args_mode != 'evaluate' and self.context_2d == 'unaligned':
            if not self.addlabel_words:
                assert (mmt_results['mmt_seq_output'].shape[1] == (
                        self.text_length + obj_num * 2))  # 3D + 2D = obj_num * 2
            else:
                assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + 2 + obj_num * 3))
        if self.pretrain:
            result["mlm_pred"] = self.mlm_cls(mmt_results['mmt_txt_output'])
            result["contra_pred"] = self.contra_cls(mmt_results['mmt_seq_output'][:, 0, :])

        result['logits'] = self.matching_cls(mmt_results['mmt_obj_output'])

        # result['logits'] = self.matching_cls(torch.cat((mmt_results['mmt_obj_output'],mmt_results['mmt_seq_output'][:,0,:].unsqueeze(1).repeat(1,mmt_results['mmt_obj_output'].shape[1],1)),dim=-1))
        # result['logits'] = torch.bmm(mmt_results['mmt_obj_output'],mmt_results['mmt_seq_output'][:,0,:].unsqueeze(2)).squeeze(2)
        # result['mmt_texttoken_output'] = mmt_results['mmt_txt_output'][:,-(obj_num+1):-1,:]
        if self.loss_proj:
            if self.addlabel_words:
                result['mmt_texttoken2d_output'] = self.fw_text2dfeat(
                    mmt_results['mmt_txt_output'][:, -(obj_num + 1):-1, :])
                result['mmt_texttoken3d_output'] = self.fw_text3dfeat(
                    mmt_results['mmt_txt_output'][:, -(obj_num + 1):-1, :])
            result['mmt_obj_output'] = self.fw_3dfeat(mmt_results['mmt_obj_output'])
        else:
            if self.addlabel_words:
                result['mmt_texttoken2d_output'] = mmt_results['mmt_txt_output'][:, -(obj_num + 1):-1, :]
                result['mmt_texttoken3d_output'] = mmt_results['mmt_txt_output'][:, -(obj_num + 1):-1, :]
            result['mmt_obj_output'] = mmt_results['mmt_obj_output']
        if self.context_2d == 'unaligned':
            result['logits_2D'] = self.matching_cls_2D(mmt_results['mmt_obj_output_2D'])
            # result['logits_2D'] = self.matching_cls_2D(torch.cat((mmt_results['mmt_obj_output_2D'],mmt_results['mmt_seq_output'][:,0,:].unsqueeze(1).repeat(1,mmt_results['mmt_obj_output_2D'].shape[1],1)),dim=-1))
            # result['logits_2D'] = torch.bmm(mmt_results['mmt_obj_output_2D'],mmt_results['mmt_seq_output'][:,0,:].unsqueeze(2)).squeeze(2)
            if self.loss_proj:
                result['mmt_obj_output_2D'] = self.fw_2dfeat(mmt_results['mmt_obj_output_2D'])
            else:
                result['mmt_obj_output_2D'] = mmt_results['mmt_obj_output_2D']
        return result


def instantiate_referit3d_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim
    mmt_out_dim = args.mmt_latent_dim

    # make an object (segment) encoder for point-clouds with color
    if args.object_encoder == 'pnet_pp':
        object_encoder = single_object_encoder(geo_out_dim)
    else:
        raise ValueError('Unknown object point cloud encoder!')

    # Optional, make a bbox encoder
    object_clf = None
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)

    # if args.model.startswith('referIt3DNet'):  # baseline method
    #     # make a language encoder
    #     lang_encoder = token_encoder(vocab=vocab,
    #                                  word_embedding_dim=args.word_embedding_dim,
    #                                  lstm_n_hidden=lang_out_dim,
    #                                  word_dropout=args.word_dropout,
    #                                  random_seed=args.random_seed)
    #
    #     language_clf = None
    #     if args.lang_cls_alpha > 0:
    #         print('Adding a text-classification loss.')
    #         language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)
    #         # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.
    #
    #     # we will use a DGCNN.
    #     print('Instantiating a classic DGCNN')
    #
    #     graph_in_dim = geo_out_dim
    #     obj_lang_clf_in_dim = args.graph_out_dim
    #
    #     if args.language_fusion in ['before', 'both']:
    #         graph_in_dim += lang_out_dim
    #
    #     if args.language_fusion in ['after', 'both', 'all']:
    #         obj_lang_clf_in_dim += lang_out_dim
    #
    #     graph_encoder = DGCNN(initial_dim=graph_in_dim,
    #                           out_dim=args.graph_out_dim,
    #                           k_neighbors=args.knn,
    #                           intermediate_feat_dim=args.dgcnn_intermediate_feat_dim,
    #                           subtract_from_self=True)
    #
    #     object_language_clf = object_lang_clf(obj_lang_clf_in_dim)
    #
    #     model = ReferIt3DNet(
    #         args=args,
    #         object_encoder=object_encoder,
    #         language_encoder=lang_encoder,
    #         graph_encoder=graph_encoder,
    #         object_clf=object_clf,
    #         language_clf=language_clf,
    #         object_language_clf=object_language_clf)
    #
    # elif args.model.startswith('mmt') and args.transformer:
    if args.model.startswith('mmt') and args.transformer:
        print('Instantiating a MMT')
        lang_out_dim = 768

        language_clf = None
        if args.lang_cls_alpha > 0:
            print('Adding a text-classification loss.')
            language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)
            # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.

        # graph_in_dim = geo_out_dim
        # obj_lang_clf_in_dim = args.graph_out_dim

        # if args.language_fusion in ['before', 'both']:
        #     graph_in_dim += lang_out_dim

        # if args.language_fusion in ['after', 'both', 'all']:
        #     obj_lang_clf_in_dim += lang_out_dim

        # graph_encoder = DGCNN(initial_dim=graph_in_dim,
        #                       out_dim=args.graph_out_dim,
        #                       k_neighbors=args.knn,
        #                       intermediate_feat_dim=args.dgcnn_intermediate_feat_dim,
        #                       subtract_from_self=True)

        # object_language_clf = object_lang_clf(obj_lang_clf_in_dim)
        model = MMT_ReferIt3DNet(
            args=args,
            num_class=n_obj_classes,
            object_encoder=object_encoder,
            object_clf=object_clf,
            language_clf=language_clf,
            visudim=geo_out_dim,
            TEXT_BERT_HIDDEN_SIZE=lang_out_dim,
            MMT_HIDDEN_SIZE=mmt_out_dim,
            pretrain=args.pretrain,
            context_2d=args.context_2d,
            feat2dtype=args.feat2d,
            mmt_mask=args.mmt_mask)
    else:
        raise NotImplementedError('Unknown listener model is requested.')

    return model


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    """
    0在PAD住的位置上，non_pad的规则：[0, 1, 2, ..., max_num] < nums
    """
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
