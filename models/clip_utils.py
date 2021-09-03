import clip
import torch
from torch import nn
from loguru import logger
from models.utils import set_param, set_embed_or_standalone_layer, set_layer

"""
是直接换掉encoder还是loss辅助学习？
"""


class OnlineCLIP(nn.Module):
    def __init__(self, args, pretrained_clip=True, device=None):
        """
        pretrained_clip is True when training the entire pipeline from scratch
        """
        super(OnlineCLIP, self).__init__()
        assert args.clip_backbone in clip.available_models()
        self.args = args
        model, _ = clip.load(args.clip_backbone, device='cpu')  # jit False by default, load in CPU first

        self.dtype = model.visual.conv1.weight.dtype  # cache dtype

        self.image_resolution = model.visual.input_resolution  # for DataLoader
        self.image_feat_size = model.visual.output_dim

        # we do not need the text_projection layer, delete it
        # if not args.init_language and hasattr(model, 'text_projection'):
        #     del model.text_projection
        # 2021-08-23 21:03:01 保留text_projection去做lang_clf！

        if not pretrained_clip:  # if resume training, we need init CLIP model (it will also be overwritten in load_state_dict(), just for insurance)
            model.initialize_parameters()

        if not args.use_clip_visual:  # delete visual branch of CLIP
            model.visual = nn.Identity()

        if not args.use_clip_language:  # delete language branch of CLIP
            model.transformer = nn.Identity()
            model.token_embedding = nn.Identity()
            model.ln_final = nn.Identity()
            model.positional_embedding = nn.Identity()

        else:  # tell using direct EOS or text_projection
            logger.info("CLIP Language Enabled, Using {} as lang_clf logits...".format(
                "Direct EOS" if args.direct_eos else "Text Projection"))

        self.model = model
        if device is not None:
            self.model = self.model.to(device)

    def encode_image(self, image):  # should be unit-norm if used for contrastive learning
        if not self.args.use_clip_visual:
            raise RuntimeError("clip visual branch not enabled!")
        return self.model.encode_image(image)

    def encode_text(self, text):
        if not self.args.use_clip_language:
            raise RuntimeError("clip language branch not enabled!")
        # return self.model.encode_text(text) -> CLIP default encode_text will shrink context_len dim by the eot embedding

        # overriding default encode_text:
        x = self.model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.type(self.dtype)
        # 2021-08-20 00:11 embedding align with TextBERT
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        return x

    def classify_text(self, text, text_embedding):  # output text classifier logits following CLIP-style
        if not self.args.use_clip_language:
            raise RuntimeError("clip language branch not enabled!")
        x = text_embedding[torch.arange(text_embedding.shape[0]), text.argmax(dim=-1)]
        if not self.args.direct_eos:
            x = x @ self.model.text_projection
        # the last [EOS] -> feature representation, decide whether using a projection
        return x

    def freeze_text(self):
        self.model.transformer.eval()
        set_layer(self.model.transformer, requires_grad=False)

        self.model.token_embedding.eval()
        set_embed_or_standalone_layer(self.model.token_embedding, requires_grad=False)

        # self.model.positional_embedding.eval()   # only a parameter, no eval() method
        set_param(self.model.positional_embedding, requires_grad=False)

        self.model.ln_final.eval()
        set_embed_or_standalone_layer(self.model.ln_final, requires_grad=False)

    def unfreeze_text(self):
        self.model.transformer.train()
        set_layer(self.model.transformer, requires_grad=True)

        self.model.token_embedding.train()
        set_embed_or_standalone_layer(self.model.token_embedding, requires_grad=True)

        # self.model.positional_embedding.train()
        set_param(self.model.positional_embedding, requires_grad=True)

        self.model.ln_final.train()
        set_embed_or_standalone_layer(self.model.ln_final, requires_grad=True)

    # @property
    # def dtype(self):
    #     return self.model.visual.conv1.weight.dtype

    def forward(self, x):
        raise NotImplementedError("Do not call OnlineCLIP directly, use encode_image or encode_text!")


if __name__ == '__main__':
    # see some keys here
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            import argparse
            args = argparse.Namespace()
            args.use_clip_visual = args.use_clip_language = args.direct_eos = True
            args.clip_backbone = 'RN50x16'
            self.clip_model = OnlineCLIP(args, pretrained_clip=True, device='cpu')

    model = MyModel()
    # for name, param in model.named_parameters():
    #     print(name)
    model.clip_model.freeze_text()
    for name, param in model.named_parameters():
        print('name: {}, requires_grad: {}'.format(name, param.requires_grad))
