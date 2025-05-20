import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_new import StoConv2d, StoLinear, StoLayer, StoModel

class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

    Basic usage:
    1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
    2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
        self.bench = SparseSpeedupBench()
        self.conv_layer1 = nn.Conv2(3, 96, 3)

        if self.bench is not None:
            outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
        else:
            outputs = self.conv_layer1(inputs)
    3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """
    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data!=0.0).sum().item()/x.numel()

    def print_weights(self, w, layer):
        # w dims: out, in, k1, k2
        #outers = []
        #for outer in range(w.shape[0]):
        #    inners = []
        #    for inner in range(w.shape[1]):
        #        n = np.prod(w.shape[2:])
        #        density = (w[outer, inner, :, :] != 0.0).sum().item() / n
        #        #print(density, w[outer, inner])
        #        inners.append(density)
        #    outers.append([np.mean(inners), np.std(inner)])
        #print(outers)
        #print(w.shape, (w!=0.0).sum().item()/w.numel())
        pass

    def forward(self, layer, x, layer_id):
        if self.layer_0_idx is None: self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx: self.iter_idx += 1
        self.print_weights(layer.weight.data, layer)

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in/float(num_channels_in*batch_size)
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end)/1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        channel_sparsity_weight = sparse_channels/float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.layer_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(time_taken_s*(1.0-channel_sparsity_weight)*(1.0-channel_sparsity_input))
        self.layer_timings_sparse[layer_id].append(time_taken_s*input_sparsity*weight_sparsity)

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print('\n')
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_sparse += t_sparse
            total_time_channel_sparse += t_channel_sparse

            print('Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}'.format(layer_id, t_dense, t_channel_sparse, t_sparse))
        self.total_timings.append(total_time_dense)
        self.total_timings_sparse.append(total_time_sparse)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)

        print('Speedups for this segment:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_channel_sparse, total_time_dense/total_time_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_sparse, total_time_dense/total_time_sparse))
        print('\n')

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print('Speedups for entire training:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_channel_sparse, total_dense/total_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_sparse, total_dense/total_sparse))
        print('\n')

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)

################################################################################################
class Sto_RN_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 same_noise=False, save_features=False, bench=None):
        super().__init__()
        self.conv1 = StoConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
            use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)

        return out

# class Sto_RN_Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, \
#                  use_bnn=False, prior_mean=0, prior_std=1, \
#                  same_noise=False, save_features=False, bench=None):
#         super().__init__()
#         self.conv1 = StoConv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, \
#                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
#                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = StoConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, \
#                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.downsample = downsample

#         self.feats = []
#         self.densities = []
#         self.save_features = save_features
#         self.bench = bench

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = F.relu(out)

#         return out

class Sto_ResNet(nn.Module, StoModel):
    def __init__(self, depth, num_classes=1000, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 same_noise=False, save_features=False, bench_model=False):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Sto_RN_BasicBlock #if depth >=44 else Sto_RN_BasicBlock

        self.inplanes = 16
        self.save_features = save_features
        
        self.conv1 = StoConv2d(3, 16, kernel_size=3, padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.bn1 = nn.BatchNorm2d(16)
        self.bench = None if not bench_model else SparseSpeedupBench()
        self.layer1 = self._make_layer(block, 16, n, stride=1, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.layer2 = self._make_layer(block, 32, n, stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.layer3 = self._make_layer(block, 64, n, stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = StoLinear(64 * block.expansion, num_classes, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)

    def _make_layer(self, block, planes, blocks, stride=1, \
                    use_bnn=False, prior_mean=0, prior_std=1, same_noise=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                StoConv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False, \
                          use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise),
                nn.BatchNorm2d(block.expansion * planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                same_noise=same_noise, save_features=self.save_features, bench=self.bench))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                same_noise=same_noise, save_features=self.save_features, bench=self.bench))

        return nn.Sequential(*layers)
    
    def set_test_mean(self, test_with_mean):
        for m in self.modules():
            if isinstance(m, StoLayer):
                m.test_with_mean = test_with_mean

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
    

################################################################################################

class StoWideResNet(nn.Module, StoModel):
    """Wide Residual Network with varying depth and width.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, depth, widen_factor, num_classes=10, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 same_noise=False, save_features=False, bench_model=False):
        super(StoWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = StoBasicBlock
        # 1st conv before any network block
        self.conv1 = StoConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.bench = None if not bench_model else SparseSpeedupBench()
        # 1st block
        self.block1 = StoNetworkBlock(n, nChannels[0], nChannels[1], block, 1, \
                                        use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                        same_noise=same_noise, save_features=save_features, bench=self.bench)
        # 2nd block
        self.block2 = StoNetworkBlock(n, nChannels[1], nChannels[2], block, 2, \
                                        use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                        same_noise=same_noise, save_features=save_features, bench=self.bench)
        # 3rd block
        self.block3 = StoNetworkBlock(n, nChannels[2], nChannels[3], block, 2, \
                                        use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                        same_noise=same_noise, save_features=save_features, bench=self.bench)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = StoLinear(nChannels[3], num_classes, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.nChannels = nChannels[3]
        self.feats = []
        self.densities = []
        self.save_features = save_features

    def set_test_mean(self, test_with_mean):
        for m in self.modules():
            if isinstance(m, StoLayer):
                m.test_with_mean = test_with_mean

    def forward(self, x):
        if self.bench is not None:
            out = self.bench.forward(self.conv1, x, 'conv1')
        else:
            out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class StoBasicBlock(nn.Module):
    """Wide Residual Network basic block

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, in_planes, out_planes, stride, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 same_noise=False, save_features=False, bench=None):
        super(StoBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = StoConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = StoConv2d(out_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise) or None
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.in_planes = in_planes

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            if self.save_features:
                self.feats.append(x.clone().detach())
                self.densities.append((x.data != 0.0).sum().item()/x.numel())
        else:
            out = self.relu1(self.bn1(x))
            if self.save_features:
                self.feats.append(out.clone().detach())
                self.densities.append((out.data != 0.0).sum().item()/out.numel())
        if self.bench:
            out0 = self.bench.forward(self.conv1, (out if self.equalInOut else x), str(self.in_planes) + '.conv1')
        else:
            out0 = self.conv1(out if self.equalInOut else x)

        out = self.relu2(self.bn2(out0))
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item()/out.numel())
        if self.bench:
            out = self.bench.forward(self.conv2, out, str(self.in_planes) + '.conv2')
        else:
            out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class StoNetworkBlock(nn.Module):
    """Wide Residual Network network block which holds basic blocks.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                same_noise=False, save_features=False, bench=None):
        super(StoNetworkBlock, self).__init__()
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, \
                                      use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, same_noise=same_noise)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, \
                    use_bnn=False, prior_mean=0, prior_std=1, same_noise=False):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, \
                 use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                 same_noise=same_noise, save_features=self.save_features, bench=self.bench))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
            if self.save_features:
                self.feats += layer.feats
                self.densities += layer.densities
                del layer.feats[:]
                del layer.densities[:]
        return x