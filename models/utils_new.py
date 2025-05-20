import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import torch.nn.init as init
import math

class StoLayer(object):
    def sto_init(self, prior_mean, prior_std):
        self.posterior_std = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        
        init.kaiming_uniform_(self.weight, nonlinearity='relu')
        init.constant_(self.posterior_std, -6.)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        self.test_with_mean = False

        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)

    def kl(self):
        prior = D.Normal(self.prior_mean, self.prior_std)
        posterior = D.Normal(self.weight, F.softplus(self.posterior_std))
        kl = (D.kl_divergence(posterior, prior) * (self.posterior_std != 0)).sum()
        return kl

class StoModel(object):
    def kl(self):
        kl = 0
        for m in self.modules():
            if isinstance(m, StoLayer):
                kl += m.kl()
        return kl


class StoConv2d(nn.Conv2d, StoLayer):
    def __init__(self, in_planes, out_planes, kernel_size, \
                use_bnn, prior_mean, prior_std, same_noise, \
                stride=1, padding=0, groups=1, bias=True, dilation=1):
        super(StoConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.use_bnn = use_bnn
        self.same_noise = same_noise
        self.input_size = None
        if not self.use_bnn:
            self.same_noise = False
        if self.use_bnn:
            self.sto_init(prior_mean, prior_std)
    
    def forward(self, x):
        self.input_size = x.size()
        if not self.same_noise:
            mean = super()._conv_forward(x, self.weight, None)
            if not self.use_bnn or self.test_with_mean:
                return mean
            else:
                std = torch.sqrt(super()._conv_forward(x**2, F.softplus(self.posterior_std)**2 * (self.posterior_std != 0), None) + 1e-8)
                
                return mean + std * torch.randn_like(mean)
        else:
            super()._conv_forward(x, self.weight + torch.randn_like(self.weight) * F.softplus(self.posterior_std) * (self.posterior_std != 0), None)

class StoLinear(nn.Linear, StoLayer):
    def __init__(self, in_features, out_features, \
                 use_bnn, prior_mean, prior_std, same_noise, bias=True):
        super(StoLinear, self).__init__(in_features, out_features, bias)
        self.use_bnn = use_bnn
        self.same_noise = same_noise
        if not self.use_bnn:
            self.same_noise = False
        if self.use_bnn:
            self.sto_init(prior_mean, prior_std)
    
    def forward(self, x):
        if not self.same_noise:
            mean = super().forward(x)
            if not self.use_bnn or self.test_with_mean:
                return mean
            else:
                std = torch.sqrt(F.linear(x**2, F.softplus(self.posterior_std)**2 * (self.posterior_std != 0)) + 1e-8)
                
                return mean + std * torch.randn_like(mean)
        else:
            F.linear(x, self.weight + torch.randn_like(self.weight) * F.softplus(self.posterior_std) * (self.posterior_std != 0), self.bias)
            

def bnn_sample(model, args):
    for n, p in model.named_parameters():
        if n.endswith('weight') and n.replace('weight', 'posterior_std') in model.state_dict():
            posterior_std = model.state_dict()[n.replace('weight', 'posterior_std')].data
            assert ((p == 0) != (posterior_std == 0)).sum() == 0
            p.data = p.data + torch.randn_like(p.data) * F.softplus(posterior_std) * (posterior_std != 0)

    return model