import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)


class TextBert(BertPreTrainedModel):
    def __init__(self, config, mmt_mask=None, addlabel_words=False):
        super().__init__(config)

        self.mmt_mask = mmt_mask
        self.addlabel_words = addlabel_words
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask, txt_type_mask=None):
        ## https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_bert.html
        encoder_inputs = self.embeddings(txt_inds, token_type_ids=txt_type_mask)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.mmt_mask == 'train2dmasklabel':
            to_seq_length = attention_mask.size(1)
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )
            # decoding step elements can attend to themselves in a causal manner
            num_query = 24
            if self.addlabel_words:
                extended_attention_mask[:, :, :num_query, num_query:] = 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]
        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config, context_2d=None, mmt_mask=None, addlabel_words=False):
        super().__init__(config)

        self.context_2d = context_2d
        self.mmt_mask = mmt_mask
        self.addlabel_words = addlabel_words
        # self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # # if self.mmt_type_emd:
        # if True:    ## img/text type emd
        #     self.type_embeddings = nn.Embedding(2, config.hidden_size)
        #     self.LayerNorm = BertLayerNorm(config.hidden_size)
        #     self.dropout = nn.Dropout(0.1)
        # # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_emb, txt_mask, obj_emb, obj_mask, obj_num):
        # scene_context = torch.max(obj_emb * obj_mask.unsqueeze(-1).repeat(1,1,obj_emb.shape[-1]),dim=1)[0]
        # obj_emb = torch.cat([scene_context.unsqueeze(1),obj_emb],dim=1)
        # obj_mask = torch.cat([obj_mask[:,0].unsqueeze(1),obj_mask],dim=1)

        encoder_inputs = torch.cat([txt_emb, obj_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask], dim=1)  # N, obj_len+txt_len

        # type_idx = torch.ones(encoder_inputs[:,:,0].shape).long().to(txt_emb.device)
        # type_idx[:,:txt_emb.shape[1]] = 0
        # encoder_inputs = self.dropout(self.LayerNorm(encoder_inputs + self.type_embeddings(type_idx)))
        ## as in M4C, do LayerNorm+Dropout at each input xxx_emb, and no pos/type emd
        encoder_inputs = encoder_inputs

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        obj_begin = txt_max_num

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # N, 1, 1, obj_len+txt_len
        ## special attention masks over lang, 3d, 2d
        if self.mmt_mask == 'train2d':
            # [batch_size, from_seq_length, to_seq_length]
            # mask type 1: 3d, lang can't see 2d
            # 对应SAT training
            to_seq_length = attention_mask.size(1)  # obj_len+txt_len
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )  # N, 1, obj_len+txt_len (from), obj_len+txt_len (to)
            # 在from上的顺序应该是：text, 3D, 2D
            # decoding step elements can attend to themselves in a causal manner
            num_2d = obj_max_num // 2
            extended_attention_mask[:, :, :-num_2d, -num_2d:] = 0.

        elif self.mmt_mask == 'train2dmasklabel':
            to_seq_length = attention_mask.size(1)
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )
            # decoding step elements can attend to themselves in a causal manner
            num_2d = obj_max_num // 2  ## (24+2+obj_num*3)
            num_query = 24
            if not self.addlabel_words:
                extended_attention_mask[:, :, :-num_2d, -num_2d:] = 0.  # 与train2d一致
                # extended_attention_mask[:, :, -num_2d:, :-num_2d] = 0.    ## ablation: mask B
            else:
                # extended_attention_mask[:, :, :-num_2d, -num_2d:] = 0.
                extended_attention_mask[:, :, :num_query, num_query:-obj_max_num] = 0.
                extended_attention_mask[:, :, :num_query, -num_2d:] = 0.
                extended_attention_mask[:, :, -obj_max_num:-num_2d, num_query:-obj_max_num] = 0.
                extended_attention_mask[:, :, -obj_max_num:-num_2d, -num_2d:] = 0.
                # extended_attention_mask[:, :, num_query:-obj_max_num, -num_2d:] = 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_max_num]
        mmt_obj_output = mmt_seq_output[:, txt_max_num:]
        # mmt_obj_output = mmt_seq_output[:, txt_max_num+1:]
        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_obj_output': mmt_obj_output[:, :obj_num, :],  # 3D
        }
        if self.context_2d == 'unaligned':
            results['mmt_obj_output_2D'] = mmt_obj_output[:, obj_num:, :]  # 2D
        return results


class MatchingLinear(nn.Module):
    def __init__(self, input_size=192, hidden_size=128, outputdim=1):
        super(MatchingLinear, self).__init__()
        hidden_size = input_size * 2 // 3
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, outputdim)

    def forward(self, x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state).squeeze(2)


"""
From VilBert, vilbert/vilbert
"""


class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights, input_size=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(input_size=input_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, input_size=None):
        super(BertPredictionHeadTransform, self).__init__()
        hidden_act = "gelu"
        hidden_size = 768
        if input_size is None:
            input_size = hidden_size
        ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.dense = nn.Linear(input_size, hidden_size)
        # if isinstance(hidden_act, str) or (
        #         sys.version_info[0] == 2 and isinstance(hidden_act, unicode)
        # ):
        self.transform_act_fn = ACT2FN[hidden_act]
        # else:
        #     self.transform_act_fn = hidden_act
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PolluteLinear(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(PolluteLinear, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state)


# class MMTMatchModule(nn.Module):
#     """
#     Input:
#         Visual: data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
#                 data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1
#         Text: data_dict['token_inds']
#               data_dict['token_num']
#     Output:
#         Matching score: data_dict["cluster_ref"]
#         Lang CLS score: data_dict["lang_scores"]
#     """
#     def __init__(self, num_class, num_proposals=256, lang_size=256, visudim=128, TEXT_BERT_HIDDEN_SIZE=768, MMT_HIDDEN_SIZE=192):
#         super(MMTMatchModule, self).__init__()
#         ## https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_bert.html
#         ## class BertConfig(PretrainedConfig)
#         self.text_bert_config = BertConfig(
#                  hidden_size=TEXT_BERT_HIDDEN_SIZE,
#                  num_hidden_layers=3,
#                  num_attention_heads=12,
#                  type_vocab_size=2)
#         self.text_bert = TextBert.from_pretrained(
#             'bert-base-uncased', config=self.text_bert_config
#         )
#         self.text_bert_out_linear = nn.Linear(
#             TEXT_BERT_HIDDEN_SIZE, MMT_HIDDEN_SIZE
#         )
#         self.linear_obj_feat_to_mmt_in = nn.Linear(
#             visudim, MMT_HIDDEN_SIZE
#         )
#         # self.obj_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
#         # self.obj_drop = nn.Dropout(0.1)

#         self.mmt_config = BertConfig(
#                  hidden_size=MMT_HIDDEN_SIZE,
#                  num_hidden_layers=4,
#                  num_attention_heads=12,
#                  type_vocab_size=2)
#         self.mmt = MMT(self.mmt_config)
#         # self.mmt = MMT.from_pretrained(
#         #     'bert-base-uncased', config=self.mmt_config
#         # )
#         self.lang_cls = nn.Sequential(
#             nn.Linear(MMT_HIDDEN_SIZE, num_class),
#             # nn.Dropout()
#         )
#         self.matching_cls = MatchingLinear(input_size=MMT_HIDDEN_SIZE)

#     def forward(self, data_dict):
#         # unpack outputs from detection branch
#         obj_feat = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
#         coords = data_dict['aggregated_vote_xyz'] # batch_size, num_proposal, 128
#         objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1
#         # obj_mmt_in = self.obj_drop(self.obj_feat_layer_norm(
#         #         self.linear_obj_feat_to_mmt_in(obj_feat)))
#         obj_mmt_in = self.linear_obj_feat_to_mmt_in(obj_feat)
#         obj_mask = _get_mask(data_dict['num_bbox'].to(obj_mmt_in.device), obj_mmt_in.size(1))    ## all proposals are non-empty

#         # unpack outputs from language branch
#         txt_inds = data_dict["token_inds"] # batch_size, lang_size
#         txt_mask = _get_mask(data_dict['token_num'].to(txt_inds.device), txt_inds.size(1))  ## all proposals are non-empty
#         text_bert_out = self.text_bert(
#             txt_inds=txt_inds,
#             txt_mask=txt_mask
#         )
#         txt_emb = self.text_bert_out_linear(text_bert_out)

#         mmt_results = self.mmt(
#             txt_emb=txt_emb,
#             txt_mask=txt_mask,
#             obj_emb=obj_mmt_in,
#             obj_mask=obj_mask
#         )
#         # print(txt_inds.shape,txt_emb.shape,obj_mmt_in.shape,mmt_results['mmt_seq_output'].shape)
#         # torch.Size([8, 128]) torch.Size([8, 128, 192]) torch.Size([8, 256, 192]) torch.Size([8, 384, 192])

#         data_dict["lang_scores"] = self.lang_cls(txt_emb[:,0,:])
#         data_dict["cluster_ref"] = self.matching_cls(mmt_results['mmt_obj_output'])
#         return data_dict

## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
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


def swish(x):
    return x * torch.sigmoid(x)
