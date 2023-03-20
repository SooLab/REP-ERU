from collections import OrderedDict
import math
import random
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
from torch.nn.init import kaiming_normal, kaiming_uniform_
from .darknet import ConvBatchNormReLU, ConvBatchNormReLU_3d

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.spacial_dim, self.embed_dim)[:H, :W]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map
           
class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, att_level3=False, baseline=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(4, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.reg = torch.nn.Sequential(
                        nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
                        nn.Sigmoid()
        )
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        self.att_level3 = att_level3
        self.baseline = baseline

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        out1 = self.reg(x)
        outs.append(out1)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])
        if self.att_level3:
            new_outs = [outs[0], outs[1], outs[2], outs[4][1], outs[4]]
            if self.baseline:
                new_outs = new_outs[:-1]
            return tuple(new_outs)
        else:
            return tuple(outs)

class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        #print(x.shape)
        #exit()
        x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        #x = self.out_proj(x)
        return x

class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=3, heads=2, output_dim=512, out_indices=[0,1,2], pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.out_indices = out_indices

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.GroupNorm(1, embed_dim)

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.GroupNorm(1, embed_dim)

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
    

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            #print(u[0])
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        #x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x
        #print(x.shape)
        B, C, H, W = x.shape

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]


        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
  
        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x[1:,: , :].permute(1, 2, 0).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform_
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas

def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
    if len(attn_score.shape)!=2:
        attn_score = attn_score.squeeze(2).squeeze(2)
    word_mask_cp = word_mask[:,:attn_score.shape[1]].clone()
    score = F.softmax(attn_score*tempuature, dim=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii,word_mask_cp[ii,:].sum()-1]=0
            else:
                word_mask_cp[ii,0]=0
                word_mask_cp[ii,word_mask_cp[ii,:].sum()]=0 ## set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score/(mask_score.sum(1)+1e-8).view(mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
    return mask_score

class FiLMedConvBlock_context(nn.Module):
    def __init__(self, with_residual=True, with_batchnorm=True,
                             with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                             with_input_proj=1, num_cond_maps=8, kernel_size=1, batchnorm_affine=False,
                             num_layers=1, condition_method='bn-film', debug_every=float('inf'),
                             textdim=768,visudim=512,contextdim=512,emb_size=512,fusion='prod',cont_map=False,
                             lstm=False,baseline=False):
        super(FiLMedConvBlock_context, self).__init__()

        self.cont_map = cont_map    ## mapping context with language feature
        self.lstm = lstm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.fusion = fusion
        self.baseline = baseline
        self.film = FiLM()

        if self.cont_map:
            self.sent_map = nn.Linear(768, emb_size)
            self.context_map = nn.Linear(emb_size, emb_size)
        if self.fusion == 'cat':
            self.attn_map = nn.Conv1d(textdim+visudim, emb_size//2, kernel_size=1)
        elif self.fusion == 'prod':
            assert(textdim==visudim) ## if product fusion
            self.attn_map = nn.Conv1d(visudim, emb_size//2, kernel_size=1)

        self.attn_score = nn.Conv1d(emb_size//2, 1, kernel_size=1)
        if self.baseline:
            self.fusion_layer = ConvBatchNormReLU(visudim+textdim+8, emb_size, 1, 1, 0, 1)
        else:
            self.gamme_decode = nn.Linear(textdim, 2 * emb_size)
            self.conv1 = nn.Conv2d(visudim+8, emb_size, kernel_size=1)
            # self.bn1 = nn.BatchNorm2d(emb_size)
            self.bn1 = nn.InstanceNorm2d(emb_size)
        init_modules(self.modules())


    def forward(self, fvisu, fword, context_score, fcoord,gest, textattn=None,weight=None,fsent=None,word_mask=None):
        fword = fword.permute(0, 2, 1)
        B, Dvisu, H, W = fvisu.size()
        B, Dlang, N = fword.size()
        B, N = context_score.size()
        assert(Dvisu==Dlang)

        if self.cont_map and fsent is not None:
            fsent = F.normalize(F.relu(self.sent_map(fsent)), p=2, dim=1)
            fcont = torch.matmul(context_score.view(B,1,N),fword.permute(0,2,1)).squeeze(1)
            fcontext = F.relu(self.context_map(fsent*fcont)).unsqueeze(2).repeat(1,1,N)
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword, fcontext], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * \
                    fword * fcontext
        else:
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * \
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)
                #print(context_tile.shape)
                #print(tile_visu.shape)
                
        attn_feat = F.tanh(self.attn_map(context_tile))
        attn_score = self.attn_score(attn_feat).squeeze(1)
        mask_score = mask_softmax(attn_score,word_mask,lstm=self.lstm)
        attn_lang = torch.matmul(mask_score.view(B,1,N),fword.permute(0,2,1))
        attn_lang = attn_lang.view(B,Dlang).squeeze(1)

        if self.baseline:
            fmodu = self.fusion_layer(torch.cat([fvisu,\
                attn_lang.unsqueeze(2).unsqueeze(2).repeat(1,1,fvisu.shape[-1],fvisu.shape[-1]),fcoord],dim=1))
        else:
            ## lang-> gamma, beta
            film_param = self.gamme_decode(attn_lang)
            film_param = film_param.view(B,2*self.emb_size,1,1).repeat(1,1,H,W)
            #print(film_param.shape)
            gammas, betas = torch.split(film_param, self.emb_size, dim=1)
            
            gammas, betas = F.tanh(gammas), F.tanh(betas)
            #gest = F.tanh(gest)
            # GEST LANGUAGE FUSION
            # gammas = gammas * gest.repeat(1,512,1,1).detach()
            # betas = betas * gest.repeat(1,512,1,1).detach()

            ## modulate visu feature
            fmodu = self.bn1(self.conv1(torch.cat([fvisu,fcoord],dim=1)))
            #print(fmodu.shape)
            #print(gammas.shape)
            #print(betas.shape)
            #exit()
            fmodu = self.film(fmodu, gammas, betas)
            fmodu = F.relu(fmodu)
        if self.with_residual:
            if weight is None:
                fmodu = fvisu + fmodu
            else:
                weight = weight.view(B,1,1,1).repeat(1, Dvisu, H, W)
                fmodu = (1-weight)*fvisu + weight*fmodu
        return fmodu, attn_lang, attn_score

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class FiLMedConvBlock_multihop(nn.Module):
    def __init__(self, NFilm=2, with_residual=True, with_batchnorm=True,
                             with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                             with_input_proj=1, num_cond_maps=8, kernel_size=1, batchnorm_affine=False,
                             num_layers=1, condition_method='bn-film', debug_every=float('inf'),
                             textdim=768,visudim=512,emb_size=512,fusion='cat',intmd=False,lstm=False,erasing=0.):
        super(FiLMedConvBlock_multihop, self).__init__()

        self.NFilm = NFilm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.cont_size = emb_size
        self.fusion = fusion
        self.intmd = intmd
        self.lstm = lstm
        self.erasing = erasing
        if self.fusion=='cat':
            self.cont_size = emb_size*2

        self.modulesdict = nn.ModuleDict()
        modules = OrderedDict()
        modules["film0"] = FiLMedConvBlock_context(textdim=textdim,visudim=emb_size,contextdim=emb_size,emb_size=emb_size,fusion=fusion,lstm=self.lstm)
        for n in range(1,NFilm):
            modules["conv%d"%n] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
            modules["film%d"%n] = FiLMedConvBlock_context(textdim=textdim,visudim=emb_size,contextdim=self.cont_size,emb_size=emb_size,fusion=fusion,lstm=self.lstm)
        self.modulesdict.update(modules)

    def forward(self, fvisu, fword, fcoord,gest = None, weight=None,fsent=None,word_mask=None):
        B, Dvisu, H, W = fvisu.size()
        B, N, Dlang = fword.size()
        intmd_feat, attnscore_list = [], []

        x, _, attn_score = self.modulesdict["film0"](fvisu, fword, Variable(torch.ones(B,N).cuda()), fcoord,gest, fsent=fsent,word_mask=word_mask)
        attnscore_list.append(attn_score.view(B,N,1,1))
        if self.intmd:
            intmd_feat.append(x)
        if self.NFilm==1:
            intmd_feat = [x]
        for n in range(1,self.NFilm):
            score_list = [mask_softmax(score.squeeze(2).squeeze(2),word_mask,lstm=self.lstm) for score in attnscore_list]

            score = torch.clamp(torch.max(torch.stack(score_list, dim=1), dim=1, keepdim=False)[0],min=0.,max=1.)
            x = self.modulesdict["conv%d"%n](x)
            x, _, attn_score = self.modulesdict["film%d"%n](x, fword, (1-score), fcoord,gest, fsent=fsent,word_mask=word_mask)
            attnscore_list.append(attn_score.view(B,N,1,1)) ## format match div loss in main func
            if self.intmd:
                intmd_feat.append(x)
            elif n==self.NFilm-1:
                intmd_feat = [x]
        return intmd_feat, attnscore_list

class Vector(nn.Sequential):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=3, heads=2, output_dim=3, out_indices=[0,1,2], pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.out_indices = out_indices

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]


        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x =  self.ln_post(x[0,:,:]) @ self.proj 
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





if __name__ == "__main__":
    import torch
    import numpy as np
    
    vect = Vector()

    dep = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    paf = torch.autograd.Variable(torch.randn(1, 1, 256, 256))
    output  = model(paf)
    print(output)
    # print(output1.size(), output2.size(), output3.size())
    # print(model(image))
    # print(len(output), output[0].size(), output[1].size(), output[2].size())
