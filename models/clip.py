import clip
import torch
from torch import nn

"""
是直接换掉encoder还是loss辅助学习？
对visual来现在有：1. offline grid-feat (各种backbone) 2. online encoder(clip / detector) 3. freeze, clip L1 loss
对language来说有：1. 保持原样的TextBERT   2. clip换掉现在的预训练TextBERT，这正好跟SAT的2D Grounding形式一致 
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

        self.dtype = model.visual.conv1.weight.dtype   # cache dtype

        self.image_resolution = model.visual.input_resolution  # for DataLoader
        self.image_feat_size = model.visual.output_dim

        # we do not need the text_projection layer, delete it
        if hasattr(model, 'text_projection'):
            del model.text_projection

        if not pretrained_clip:  # if resume training, we need init CLIP model (it will also be overwritten in load_state_dict(), just for insurance)
            model.initialize_parameters()

        if not args.use_clip_visual:  # delete visual branch of CLIP
            model.visual = nn.Identity()

        if not args.use_clip_language:  # delete language branch of CLIP
            model.transformer = nn.Identity()

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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        return x

    # @property
    # def dtype(self):
    #     return self.model.visual.conv1.weight.dtype

    def forward(self, x):
        raise NotImplementedError("Do not call OnlineCLIP directly, use encode_image or encode_text!")
