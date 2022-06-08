import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _single, _pair, _triple

def sigmoid(z):
    return 1. / (1 + torch.exp(-z))

def logit(z):
    return torch.log(z/(1.-z))

def gumbel_softmax(logits, U, temp, hard=False, eps=1e-10):
    z = logits + torch.log(U + eps) - torch.log(1 - U + eps)
    y = 1 / (1 + torch.exp(- z / temp))
    if not hard:
        return y
    y_hard = (y > 0.5).float()
    y_hard = (y_hard - y).detach() + y
    return y_hard

##########################################################################################################################################
#### Spike-and-slab Gaussian batchnorm Layer
class SSGauss_VB_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_mu = nn.Parameter(torch.Tensor(num_features))
            self.weight_rho = nn.Parameter(torch.Tensor(num_features))

            self.bias_mu = nn.Parameter(torch.Tensor(num_features))
            self.bias_rho = nn.Parameter(torch.Tensor(num_features))

            self.theta = nn.Parameter(torch.Tensor(num_features))

        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_rho', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_parameter('theta', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        # initialize weight samples and binary indicators z
        self.weight = None
        self.bias = None
        self.z = None
        self.z_extra = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
        self.reset_parameters()
        # initialize kl for the hidden layer
        self.kl = 0

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight_mu)
            init.constant_(self.weight_rho, -6.)
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_rho, -6.)
            init.constant_(self.theta, logit(torch.tensor(0.99)))
            init.constant_(self.z_extra, 1)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class SSGauss_VB_BatchNorm(SSGauss_VB_NormBase):

    def __init__(self, num_features, freeze, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, temp = 0.5, gamma_prior = 0.0001, 
                 sigma_0 = 1, testing = 0):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.freeze = freeze
        self.testing = testing

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            # return F.batch_norm(
            # input, self.running_mean, self.running_var, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask),
            # self.training or not self.track_running_stats,
            # exponential_average_factor, self.eps)
            if self.freeze == 0:
                u = torch.zeros_like(self.weight_mu).uniform_(0.0, 1.0)
                self.z = gumbel_softmax(self.theta, u, self.temp, hard=True)
                self.z_extra = nn.Parameter(gumbel_softmax(self.theta, u, self.temp, hard=True), requires_grad=False)
            
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))    
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            if self.testing == 0:
                self.weight = self.z * (self.weight_mu + weight_sigma * weight_epsilon)
            else:
                self.weight = self.z_extra * (self.weight_mu + weight_sigma * weight_epsilon)

            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
            if self.testing == 0:
                self.bias = self.z * (self.bias_mu + bias_sigma * bias_epsilon)
            else:
                self.bias = self.z_extra * (self.bias_mu + bias_sigma * bias_epsilon)

            if self.training:      
                gamma = sigmoid(self.theta)
            
                kl_gamma = gamma * (torch.log(gamma) - torch.log(self.gamma_prior)) + \
                (1 - gamma) * (torch.log(1 - gamma) - torch.log(1 - self.gamma_prior)) 
            
                kl_weight = gamma * (torch.log(self.sigma_0) - torch.log(weight_sigma) +
                            0.5*(weight_sigma ** 2 + self.weight_mu ** 2)/self.sigma_0**2 - 0.5)
            
                kl_bias = gamma * (torch.log(self.sigma_0) - torch.log(bias_sigma) +
                            0.5*(bias_sigma ** 2 + self.bias_mu ** 2)/self.sigma_0**2 - 0.5)
                
                self.kl = torch.sum(kl_gamma) + torch.sum(kl_weight) + torch.sum(kl_bias)  

            return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        else:
            self.weight = None
            self.bias = None
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        
class SSGauss_VB_BatchNorm2d(SSGauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class SSGauss_VB_BatchNorm1d(SSGauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

##########################################################################################################################################
#### Gaussian batchnorm Layer
class Gauss_VB_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_mu = nn.Parameter(torch.Tensor(num_features))
            self.weight_rho = nn.Parameter(torch.Tensor(num_features))

            self.bias_mu = nn.Parameter(torch.Tensor(num_features))
            self.bias_rho = nn.Parameter(torch.Tensor(num_features))

        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_rho', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.weight = None
        self.bias = None
        # initialize kl for the hidden layer
        self.kl = 0

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight_mu)
            init.constant_(self.weight_rho, -6.)
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_rho, -6.)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class Gauss_VB_BatchNorm(Gauss_VB_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, sigma_0 =1):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            # return F.batch_norm(
            # input, self.running_mean, self.running_var, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask),
            # self.training or not self.track_running_stats,
            # exponential_average_factor, self.eps)

            weight_sigma = torch.log1p(torch.exp(self.weight_rho))    
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            self.weight = self.weight_mu + weight_sigma * weight_epsilon

            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
            self.bias = self.bias_mu + bias_sigma * bias_epsilon

            if self.training:
                kl_weight = (torch.log(self.sigma_0) - torch.log(weight_sigma) +
                        0.5*(weight_sigma ** 2 + self.weight_mu ** 2)/self.sigma_0**2 - 0.5)
        
                kl_bias = (torch.log(self.sigma_0) - torch.log(bias_sigma) +
                        0.5*(bias_sigma ** 2 + self.bias_mu ** 2)/self.sigma_0**2 - 0.5)
            
                self.kl = torch.sum(kl_weight) + torch.sum(kl_bias) 

            return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        else:
            self.weight = None
            self.bias = None
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

class Gauss_VB_BatchNorm2d(Gauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class Gauss_VB_BatchNorm1d(Gauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
