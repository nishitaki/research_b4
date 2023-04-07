import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp, LayerNorm2d, LayerNorm, \
    create_conv2d, get_act_layer, make_divisible, to_ntuple


from einops.einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Mish()
        # nn.GELU()
    )



class MixConv:
    """MixConv with mixed depth-wise convolutional kernels.
      MDConv is an improved depth-wise convolution that mixes multiple kernels (e.g.
      3x3, 5x5, etc). Right now, we use an naive implementation that split channels
      into multiple groups and perform different kernels for each group.
      See MixNet paper https://arxiv.org/abs/1907.09595 for more details."""

    def __init__(self, ksize: list, stride: int, dilated=False, bias=False):
        """
        :param ksize: list of int, kernel_size of each group;
        :param stride: int, stride for each group;
        :param dilated: bool. indicate whether to use dilated conv to simulate large
                        kernel size, note that this only take effect when stride is 1;
        :param bias: convolution layer bias.
        """

        self.k_list = []
        self.d_list = []

        for k in ksize:
            d = 1
            # Only apply dilated conv for stride 1 if needed
            if stride == 1 and dilated:
                d = (k - 1) // 2
                k = 3

            self.k_list.append(k)
            self.d_list.append(d)

        self.stride = stride
        self.bias = bias

    def __call__(self, x):
        num_groups = len(self.k_list)
        # returns a tuple
        x_split = torch.split(x, num_groups, dim=1)

        # with dilation, the truth kernel size is: (k - 1) * d + 1
        conv_list = [nn.Conv2d(x.size(1), x.size(1), k, self.stride,
                               padding=(k - 1) * d // 2, dilation=d, groups=x.size(1), bias=self.bias)
                     for k, d, x in zip(self.k_list, self.d_list, x_split)]

        out_conv = [conv(inputs) for conv, inputs in zip(conv_list, x_split)]
        out = torch.cat(out_conv, dim=1)

        return out

class FReLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)



# class EdgeResidual(nn.Module):
#     """ Residual block with expansion convolution followed by pointwise-linear w/ stride
#     Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
#         - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
#     This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
#       * MobileDet - https://arxiv.org/abs/2004.14525
#       * EfficientNet-X - https://arxiv.org/abs/2102.05610
#       * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
#     """

#     def __init__(
#             self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, group_size=0, pad_type='',
#             force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, act_layer=nn.ReLU,
#             norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
#         super(EdgeResidual, self).__init__()
#         norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
#         if force_in_chs > 0:
#             mid_chs = make_divisible(force_in_chs * exp_ratio)
#         else:
#             mid_chs = make_divisible(in_chs * exp_ratio)
#         groups = num_groups(group_size, in_chs)
#         self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

#         # Expansion convolution
#         self.conv_exp = create_conv2d(
#             in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type)
#         self.bn1 = norm_act_layer(mid_chs, inplace=True)

#         # Squeeze-and-excitation
#         self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

#         # Point-wise linear projection
#         self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
#         self.bn2 = norm_act_layer(out_chs, apply_act=False)
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

#     def feature_info(self, location):
#         if location == 'expansion':  # after SE, before PWL
#             return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
#         else:  # location == 'bottleneck', block output
#             return dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)

#     def forward(self, x):
#         shortcut = x
#         x = self.conv_exp(x)
#         x = self.bn1(x)
#         x = self.se(x)
#         x = self.conv_pwl(x)
#         x = self.bn2(x)
#         if self.has_skip:
#             x = self.drop_path(x) + shortcut
#         return x


###=============================convnext公式から=====================================================
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



##=====================自作convnext bolock====================================

class ConvNext(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Mish(),
                # nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
        
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                LayerNorm(hidden_dim),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                
                # down-sample in the first conv
                
                nn.BatchNorm2d(hidden_dim),
                nn.Mish(),
                # 
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# class PostNorm(nn.Module):
#     def __init__(self,dim,fn,norm):
#         super().__init__()
#         self.norm = norm(dim)
#         self.fn = fn

#     def forward(self, x,**kwargs):
#         return  self.norm()


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp * expansion), bias=False), ###########
            # FReLU(int(inp * expansion)),
            nn.ReLU(),
            # FReLU(int(inp * expansion)),
            # nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(),
            # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Fused_MVConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                FReLU(hidden_dim),
                # nn.Mish(),
                # nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp,hidden_dim,3,stride,1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                FReLU(hidden_dim),
                # nn.Mish(),
                # nn.GELU(),
                SE(hidden_dim, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                FReLU(oup),
                # nn.Mish(),  #MBConvのときは入れていなかった
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class MBConv_rw(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                FReLU(hidden_dim),
                # nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # LayerNorm(hidden_dim),
                FReLU(hidden_dim),
                # nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                          
                SE(hidden_dim, hidden_dim), ###################場所入れ替################
                nn.BatchNorm2d(hidden_dim),
                # LayerNorm(hidden_dim),
                # nn.GELU(),
                FReLU(hidden_dim),
                
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                # LayerNorm(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)



class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(hidden_dim, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)





##BNかLYか　正規化層について
class MBConv_imp(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Mish(),
                # nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 7, 1, 1, groups=inp, bias=False),
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Mish(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)




class Fused_MVConv_dflt(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # nn.Mish(),
                # nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp,hidden_dim,3,stride,1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # nn.Mish(),
                # nn.GELU(),
                SE(hidden_dim, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # FReLU(oup),
                nn.ReLU(),  #MBConvのときは入れていなかった
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__() 
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x



class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=4, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer,'N': ConvNext,'F': Fused_MVConv,'FD': Fused_MVConv_dflt,'C2': MBConv_rw}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)   ###########################rwはkernel_size=2
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)


    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)



# class CoAtNet_fc(nn.Module):
#     def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=4, block_types=['C', 'C', 'T', 'T']):
#         super().__init__()
#         ih, iw = image_size
#         block = {'C': MBConv, 'T': Transformer,'N': ConvNext,'F': Fused_MVConv,'C2': MBConv_imp}

#         self.s0 = self._make_layer(
#             conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
#         self.s1 = self._make_layer(
#             block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
#         self.s2 = self._make_layer(
#             block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
#         self.s3 = self._make_layer(
#             block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
#         self.s4 = self._make_layer(
#             block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

#         self.pool = nn.AvgPool2d(ih // 32, 1)
#         self.fc = nn.Linear(channels[-1], num_classes, bias=False)

#         self.fc1 = nn.Linear(channels[-1], channels[-1], bias=True)
#         self.dropout = nn.Dropout(0.2)
        
#         self.fc2 = nn.Linear(channels[-1],num_classes,bias=False)


        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels=channels[0],out_channels=channels[2],kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(in_channels=channels[2],out_channels=channels[1],kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2)       
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=channels[1],out_channels=channels[2],kernel_size=2,stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=channels[2],out_channels=channels[1],kernel_size=2,stride=2),
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#         x = self.s0(x)

#         ##CAE
#         x = self.encoder(x)
#         x = self.decoder(x)

#         x = self.s1(x)
#         x = self.s2(x)
#         x = self.s3(x)
#         x = self.s4(x)

#         x = self.pool(x).view(-1, x.shape[1])

#         x = self.fc(x)
        
#         return x

#     def _make_layer(self, block, inp, oup, depth, image_size):
#         layers = nn.ModuleList([])
#         for i in range(depth):
#             if i == 0:
#                 layers.append(block(inp, oup, image_size, downsample=True))
#             else:
#                 layers.append(block(oup, oup, image_size))
#         return nn.Sequential(*layers)





def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=4)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=4)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=4)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=4)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))
