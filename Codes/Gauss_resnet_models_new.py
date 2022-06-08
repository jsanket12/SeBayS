import torch
import torch.nn as nn
import torch.nn.init as init
# import math
import torch.nn.functional as F

from Gauss_linear_layers import Gauss_layer
from Gauss_Conv_layers import SSGauss_Node_Conv2d_layer , Gauss_Conv2d_layer
from Gauss_BN_layers import Gauss_VB_BatchNorm2d

__all__ = ['resnet']

##########################################################################################################################################
#### Spike-and-slab node selection with Gaussian: Resnet and Wide-Resnet
def SSGauss_Node_conv3x3(in_planes, out_planes, freeze, stride=1, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
    "3x3 convolution with padding"
    return SSGauss_Node_Conv2d_layer(in_planes, out_planes, freeze, kernel_size=3, stride=stride,
                     padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)

class SSGauss_Node_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, freeze, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
        super().__init__()
        self.conv1 = SSGauss_Node_conv3x3(inplanes, planes, freeze, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)
        self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv2 = SSGauss_Node_conv3x3(planes, planes, freeze, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)
        self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

# class SSGauss_Node_Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, freeze, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
#         super().__init__()
#         self.conv1 = SSGauss_Node_Conv2d_layer(inplanes, planes, freeze, kernel_size=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)
#         self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
#         self.conv2 = SSGauss_Node_Conv2d_layer(planes, planes, freeze, kernel_size=3, stride=stride, padding=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)
#         self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
#         self.conv3 = SSGauss_Node_Conv2d_layer(planes, planes * 4, freeze, kernel_size=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, testing = testing)
#         self.bn3 = Gauss_VB_BatchNorm2d(planes * 4, sigma_0=sigma_0)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = F.silu(self.bn1(self.conv1(x)))
#         out = F.silu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = F.silu(out)

#         return out

class SSGauss_Node_ResNet_Cifar(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.freeze = nn.Parameter(torch.Tensor(1), requires_grad=False)
        init.constant_(self.freeze, 0)
        self.testing = testing

        # block = SSGauss_Node_Bottleneck if depth >=44 else SSGauss_Node_BasicBlock
        block = SSGauss_Node_BasicBlock

        self.inplanes = 16
        self.sigma_0 = sigma_0
        self.temp = temp
        self.gamma_prior = gamma_prior
        
        self.conv1 = SSGauss_Node_Conv2d_layer(3, 16,  freeze = self.freeze, kernel_size=3, padding=1, 
                                                bias=False, temp=temp, gamma_prior=gamma_prior, 
                                                sigma_0=sigma_0, testing = testing)
        self.bn1 = Gauss_VB_BatchNorm2d(16, sigma_0=sigma_0)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Gauss_layer(64 * block.expansion, num_classes, sigma_0=sigma_0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SSGauss_Node_Conv2d_layer(self.inplanes, planes * block.expansion, self.freeze,
                          kernel_size=1, stride=stride, bias=False, temp=self.temp, gamma_prior=self.gamma_prior, 
                          sigma_0=self.sigma_0, testing = self.testing),
                Gauss_VB_BatchNorm2d(planes * block.expansion, sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.freeze, stride, downsample, temp=self.temp, 
                            gamma_prior=self.gamma_prior, sigma_0=self.sigma_0, testing = self.testing))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.freeze, temp=self.temp, gamma_prior=self.gamma_prior, 
                                sigma_0=self.sigma_0, testing = self.testing))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))  # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def freeze_flag(self):
        init.constant_(self.freeze, 1)

    def unfreeze_flag(self):
        init.constant_(self.freeze, 0)

##########################################################################################################################################
#### Gaussian without spike-and-slab: Resnet and Wide-Resnet
def Gauss_conv3x3(in_planes, out_planes, stride=1, sigma_0 = 1):
    "3x3 convolution with padding"
    return Gauss_Conv2d_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sigma_0=sigma_0)

class Gauss_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1):
        super().__init__()
        self.conv1 = Gauss_conv3x3(inplanes, planes, stride, sigma_0=sigma_0)
        self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv2 = Gauss_conv3x3(planes, planes, sigma_0=sigma_0)
        self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x))) #F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out) #F.silu(out)

        return out

# class Gauss_Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1):
#         super().__init__()
#         self.conv1 = Gauss_Conv2d_layer(inplanes, planes, kernel_size=1, bias=False, sigma_0=sigma_0)
#         self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
#         self.conv2 = Gauss_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, sigma_0=sigma_0)
#         self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
#         self.conv3 = Gauss_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, sigma_0=sigma_0)
#         self.bn3 = Gauss_VB_BatchNorm2d(planes * 4, sigma_0=sigma_0)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x))) #F.silu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out))) #F.silu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = F.relu(out) #F.silu(out)

#         return out

class Gauss_ResNet_Cifar(nn.Module):

    def __init__(self, depth, num_classes=1000, sigma_0 = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Gauss_Bottleneck if depth >=44 else Gauss_BasicBlock
        block = Gauss_BasicBlock

        self.inplanes = 16
        self.sigma_0 = sigma_0
        self.conv1 = Gauss_Conv2d_layer(3, 16, kernel_size=3, padding=1, bias=False, sigma_0=sigma_0)
        self.bn1 = Gauss_VB_BatchNorm2d(16, sigma_0=sigma_0)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Gauss_layer(64 * block.expansion, num_classes, sigma_0=sigma_0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Gauss_Conv2d_layer(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, sigma_0=self.sigma_0),
                Gauss_VB_BatchNorm2d(planes * block.expansion,sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) #F.silu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# class Gauss_ResNet_ImageNet(nn.Module):
#     def __init__(self, depth, num_classes=1000, sigma_0 = 1):
#         super().__init__()
#         # Model type specifies number of layers for ImageNet model
#         assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
#         n = (depth - 4) // 6

#         block = Gauss_Bottleneck if depth >=44 else Gauss_BasicBlock

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3: 
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

#     def forward(self, x):
#         return self._forward_impl(x)