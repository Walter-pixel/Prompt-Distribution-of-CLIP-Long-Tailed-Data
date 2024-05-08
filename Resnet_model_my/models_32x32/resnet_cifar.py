import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class DownsampleC(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleC, self).__init__()
    assert stride != 1 or nIn != nOut
    self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

  def forward(self, x):
    x = self.conv(x)
    return x

class DownsampleD(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleD, self).__init__()
    assert stride == 2
    self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
    self.bn   = nn.BatchNorm2d(nOut)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x



class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models_224x224/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class ResNetBasicblock_dpr(nn.Module):
    expansion = 1
    drop_path_rate = None # I added, this is the class attribute (can over-write later to assign the drop_path_rate)
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models_224x224/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock_dpr, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.drop_path = DropPath(self.drop_path_rate) # I added, direclty using class attribute "drop_path_rate"
        print(f"drop path created, drop path rate is {self.drop_path_rate}")

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        try:
            return F.relu(self.drop_path(residual) + basicblock, inplace=True)
        except:
            print(f"drop path rate object hasn't been created yet, forget to call <initialize_dpr> when initialization")


#-----------------------------------------------------------------------------------------------------------------------
#
# class CifarResNet(nn.Module):
#     """
#     ResNet optimized for the Cifar dataset, as specified in
#     https://arxiv.org/abs/1512.03385.pdf
#     """
#
#     # def __init__(self, block, depth, num_classes, drop_rate):
#     def __init__(self, block, depth, adaptor_out_dim):
#         """ Constructor
#         Args:
#           depth: number of layers.
#           num_classes: number of classes
#           base_width: base width
#         """
#         super(CifarResNet, self).__init__()
#
#         # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
#         assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
#         layer_blocks = (depth - 2) // 6
#         print('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
#
#         # self.num_classes = num_classes
#
#         self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn_1 = nn.BatchNorm2d(16)
#
#         self.inplanes = 16
#         self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
#         self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
#         self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
#         self.avgpool = nn.AvgPool2d(8)
#         self.adaptor = nn.Linear(64*block.expansion, adaptor_out_dim)
#         # self.classifier = nn.Linear(64 * block.expansion, num_classes)
#         # self.drop = nn.Dropout(p=drop_rate) # I added
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 # m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal(m.weight)
#                 m.bias.data.zero_()
#
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x, fmap=None):
#
#         x = self.conv_1_3x3(x)
#         x = F.relu(self.bn_1(x), inplace=True)
#         x = self.stage_1(x)
#         lyr1 = x
#         x = self.stage_2(x)
#         lyr2 = x
#         x = self.stage_3(x)
#         lyr3 = x
#         x = self.avgpool(x)
#         avg_p = x
#         x = x.view(x.size(0), -1)
#
#         # logit_score = self.drop(self.classifier(x))
#         #
#         # middle_all = {'lyr1': lyr1,
#         #               'lyr2': lyr2,
#         #               'lyr3': lyr3,
#         #               'avg_p': avg_p,
#         #               'flat': flat}
#         # middle_select = {}
#         #
#         # if fmap is not None:
#         #     for idx in range(len(fmap)):
#         #         name_to_grab = fmap[idx]
#         #         middle_select[name_to_grab] = middle_all[name_to_grab]
#         #     return logit_score, middle_select
#         #
#         # else:
#         #     return logit_score
#
#         feat = self.adaptor(x)
#
#         return feat











class CifarResNet_w_Adaptor(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf

    This is CIFAR resnet32 with attention pooling adaptor in order to match dimension with the text encoder

    """

    # def __init__(self, block, depth, num_classes, drop_rate):
    def __init__(self, block, depth, adaptor_out_dim):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet_w_Adaptor, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        # self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.adaptor = nn.Linear(64*block.expansion, adaptor_out_dim)
        # self.classifier = nn.Linear(64 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, fmap=None):

        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        lyr1 = x
        x = self.stage_2(x)
        lyr2 = x
        x = self.stage_3(x)
        lyr3 = x
        x = self.avgpool(x)
        avg_p = x
        x = x.view(x.size(0), -1)

        feat = self.adaptor(x)

        return feat





class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
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

        return x[0]








def resnet32_w_adpator(adaptor_out_dim):
    return CifarResNet_w_Adaptor(ResNetBasicblock, 32, adaptor_out_dim=adaptor_out_dim)


