import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import StoConv2d, StoLinear, StoLayer, StoModel

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
            

class StoBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), \
                 same_noise=False, sigma_parameterization='softplus', \
                 save_features=False, bench=None):
        super(StoBasicBlock, self).__init__()
        self.conv1 = StoConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
            use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
            same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                               posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                               same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StoConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, \
                          use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                          posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                          same_noise=same_noise, sigma_parameterization=sigma_parameterization),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), \
                 same_noise=False, sigma_parameterization='softplus', \
                 save_features=False, bench=None):
        super(StoBottleneck, self).__init__()
        self.conv1 = StoConv2d(in_planes, planes, kernel_size=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = StoConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StoConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, \
                          use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                            sigma_parameterization=sigma_parameterization),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StoResNet(nn.Module, StoModel):
    def __init__(self, block, num_blocks, num_classes=10, \
                 use_bnn=False, prior_mean=0, prior_std=1, \
                 posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), \
                 same_noise=False, sigma_parameterization='softplus', \
                 save_features=False, bench_model=False):
        super(StoResNet, self).__init__()
        self.in_planes = 64
        self.save_features = save_features
        self.conv1 = StoConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(64)
        self.bench = None if not bench_model else SparseSpeedupBench()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                       posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                       same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                       posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                       same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                       posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                       same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                       posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                       same_noise=same_noise, sigma_parameterization=sigma_parameterization)
        self.linear = StoLinear(512*block.expansion, num_classes, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                same_noise=same_noise, sigma_parameterization=sigma_parameterization)

    def _make_layer(self, block, planes, num_blocks, stride, \
                    use_bnn=False, prior_mean=0, prior_std=1, \
                    posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), \
                    same_noise=False, sigma_parameterization='softplus'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                                posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                                same_noise=same_noise, sigma_parameterization=sigma_parameterization, \
                                save_features=self.save_features, bench=self.bench))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_test_mean(self, test_with_mean):
        for m in self.modules():
            if isinstance(m, StoLayer):
                m.test_with_mean = test_with_mean

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


def StoResNet18(num_classes, use_bnn=False, prior_mean=0, prior_std=1, \
                posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), \
                same_noise=False, sigma_parameterization='softplus',
                save_features=False, bench_model=False):
    return StoResNet(StoBasicBlock, [2, 2, 2, 2], num_classes=num_classes, \
                     use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, \
                     posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, \
                     same_noise=same_noise, sigma_parameterization=sigma_parameterization, \
                     save_features=save_features, bench_model=bench_model)