import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils_new import StoConv2d, StoLinear

########################################
# Bayesian ResNet Builder
########################################

class StoResNetBuilder(object):
    def __init__(self, config, use_bnn=False, prior_mean=0, prior_std=1, same_noise=False):
        """
        Args:
            config (dict): configuration dictionary (e.g., {'conv_init': 'fan_out'})
            use_bnn (bool): whether to use Bayesian layers.
            prior_mean, prior_std, same_noise: Bayesian parameters.
        """
        self.config = config
        self.use_bnn = use_bnn
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.same_noise = same_noise

    def conv(self, kernel_size, in_planes, out_planes, stride=1):
        if kernel_size == 3:
            conv = StoConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False,
                             use_bnn=self.use_bnn, prior_mean=self.prior_mean,
                             prior_std=self.prior_std, same_noise=self.same_noise)
        elif kernel_size == 1:
            conv = StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=0, bias=False,
                             use_bnn=self.use_bnn, prior_mean=self.prior_mean,
                             prior_std=self.prior_std, same_noise=self.same_noise)
        elif kernel_size == 7:
            conv = StoConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                             padding=3, bias=False,
                             use_bnn=self.use_bnn, prior_mean=self.prior_mean,
                             prior_std=self.prior_std, same_noise=self.same_noise)
        else:
            return None

        nn.init.kaiming_normal_(conv.weight, mode=self.config['conv_init'], nonlinearity='relu')
        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        return self.conv(3, in_planes, out_planes, stride=stride)

    def conv1x1(self, in_planes, out_planes, stride=1):
        return self.conv(1, in_planes, out_planes, stride=stride)

    def conv7x7(self, in_planes, out_planes, stride=1):
        return self.conv(7, in_planes, out_planes, stride=stride)

    def batchnorm(self, planes):
        bn = nn.BatchNorm2d(planes)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)
        return bn

    def activation(self):
        return nn.ReLU(inplace=True)

########################################
# Bayesian BasicBlock and Bottleneck
########################################

class StoBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        """
        A Bayesian version of the basic residual block.
        """
        super(StoBasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class StoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        """
        A Bayesian version of the bottleneck residual block.
        """
        super(StoBottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

########################################
# Bayesian ResNet
########################################

class StoResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000):
        """
        Args:
            builder: an instance of StoResNetBuilder.
            block: block type (StoBasicBlock or StoBottleneck).
            layers: list of layer counts for each block (e.g., [2, 2, 2, 2] for ResNet-18).
            num_classes: number of output classes.
        """
        self.inplanes = 64
        super(StoResNet, self).__init__()
        self.conv1 = builder.conv7x7(3, 64, stride=2)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = StoLinear(512 * block.expansion, num_classes,
                            use_bnn=builder.use_bnn, prior_mean=builder.prior_mean,
                            prior_std=builder.prior_std, same_noise=builder.same_noise)

        # (Optional) Initialize fc biases to zero, etc.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, StoConv2d)):
                nn.init.kaiming_normal_(m.weight, mode=builder.config['conv_init'], nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(self.inplanes, planes * block.expansion, stride=stride)
            dbn = builder.batchnorm(planes * block.expansion)
            downsample = nn.Sequential(dconv, dbn)

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

########################################
# Build Function
########################################

# These dictionaries mimic your frequentist resnet_versions and resnet_configs.
# For Bayesian ResNet, you may wish to override the block definitions:
resnet_configs = {
    'classic': {
        'conv_init': 'fan_out',
    },
    'fanin': {
        'conv_init': 'fan_in',
    },
}

# Here we swap out the basic blocks for our Bayesian versions.
# (For example, for resnet18 and resnet34 use StoBasicBlock; for deeper ones use StoBottleneck.)
resnet_versions = {
    'resnet18': {
        'block': StoBasicBlock,
        'layers': [2, 2, 2, 2],
        'num_classes': 1000,
    },
    'resnet34': {
        'block': StoBasicBlock,
        'layers': [3, 4, 6, 3],
        'num_classes': 1000,
    },
    'resnet50': {
        'block': StoBottleneck,
        'layers': [3, 4, 6, 3],
        'num_classes': 1000,
    },
    'resnet101': {
        'block': StoBottleneck,
        'layers': [3, 4, 23, 3],
        'num_classes': 1000,
    },
    'resnet152': {
        'block': StoBottleneck,
        'layers': [3, 8, 36, 3],
        'num_classes': 1000,
    },
}

def build_sto_resnet(version, config, use_bnn=False, prior_mean=0, prior_std=1, same_noise=False):
    """
    Constructs a Bayesian ResNet.

    Args:
        version (str): one of the keys in resnet_versions (e.g., 'resnet18').
        config (str): one of the keys in resnet_configs (e.g., 'classic').
        use_bnn, prior_mean, prior_std, same_noise: Bayesian parameters.
    Returns:
        model: an instance of StoResNet.
    """
    version_cfg = resnet_versions[version]
    config_cfg = resnet_configs[config]

    builder = StoResNetBuilder(config_cfg, use_bnn=use_bnn,
                               prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
    print("Version: {}".format(version_cfg))
    print("Config: {}".format(config_cfg))
    model = StoResNet(builder,
                      version_cfg['block'],
                      version_cfg['layers'],
                      num_classes=version_cfg['num_classes'])
    return model