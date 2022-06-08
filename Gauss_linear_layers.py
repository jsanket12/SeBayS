import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
# pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2).type(torch.float)

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
#### Spike-and-slab node selection with Gaussian linear layer
class SSGauss_Node_layer(nn.Module):
    
    __constants__ = ['bias', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, freeze, bias=True,
                    temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.freeze = freeze
        self.testing = testing

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.theta = nn.Parameter(torch.Tensor(output_dim))  
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(output_dim))
            self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None
        self.z = None
        self.z_extra = nn.Parameter(torch.Tensor(output_dim), requires_grad=False)
        self.reset_parameters()

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
        init.constant_(self.theta, logit(torch.tensor(0.99)))
        init.constant_(self.z_extra, 1)

    def forward(self, input):
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        if self.freeze == 0:
            u = torch.zeros_like(self.theta).uniform_(0.0, 1.0)
            self.z = gumbel_softmax(self.theta, u, self.temp, hard=True)
            self.z_extra = nn.Parameter(gumbel_softmax(self.theta, u, self.temp, hard=True), requires_grad=False)
        if self.testing == 0:
            w_z = self.z.expand(self.input_dim, self.output_dim)
        else:
            w_z = self.z_extra.expand(self.input_dim, self.output_dim)
        w_z = self.z.expand(self.input_dim, self.output_dim)
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        self.w = w_z * (self.w_mu + sigma_w * epsilon_w)

        if self.v_mu is not None:
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            if self.testing == 0:
                v_z = self.z
            else:
                v_z = self.z_extra
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
        else:
            self.v = None
        
        if self.training:
            gamma = sigmoid(self.theta)
            w_gamma = gamma.expand(self.input_dim, self.output_dim)

            kl_gamma = gamma * (torch.log(gamma) - torch.log(self.gamma_prior)) + \
                        (1 - gamma) * (torch.log(1 - gamma) - torch.log(1 - self.gamma_prior)) 
            
            kl_w = w_gamma * (torch.log(self.sigma_0) - torch.log(sigma_w) +
                            0.5*(sigma_w ** 2 + self.w_mu ** 2)/self.sigma_0**2 - 0.5)
            
            if self.v_mu is not None:
                v_gamma = gamma

                kl_v = v_gamma * (torch.log(self.sigma_0) - torch.log(sigma_v) +
                        0.5*(sigma_v ** 2 + self.v_mu ** 2)/self.sigma_0**2 - 0.5)

                self.kl = torch.sum(kl_gamma) + torch.sum(kl_w) + torch.sum(kl_v)                 
            else:
                self.kl = torch.sum(kl_gamma) + torch.sum(kl_w)

        return F.linear(input, self.w.T, self.v)

##########################################################################################################################################
#### Spike-and-slab edge selection with Gaussian linear layer
class SSGauss_Edge_layer(nn.Module):
    
    __constants__ = ['bias', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, freeze, bias=True,
                    temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, testing = 0):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.freeze = freeze
        self.testing = testing

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_theta = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(output_dim))
            self.v_rho = nn.Parameter(torch.Tensor(output_dim))
            self.v_theta = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
            self.register_parameter('v_theta', None)

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None
        self.w_z = None
        self.v_z = None
        self.w_z_extra = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=False)
        if bias:            
            self.v_z_extra = nn.Parameter(torch.Tensor(output_dim), requires_grad=False)
        self.reset_parameters()

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)
        init.constant_(self.w_theta, logit(torch.tensor(0.99)))
        init.constant_(self.w_z_extra, 1)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
            init.constant_(self.v_theta, logit(torch.tensor(0.99)))
            init.constant_(self.v_z_extra, 1)

    def forward(self, input):
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        if self.freeze == 0:
            u_w = torch.zeros_like(self.w_mu).uniform_(0.0, 1.0)        
            self.w_z = gumbel_softmax(self.w_theta, u_w, self.temp, hard=True)
            self.w_z_extra = nn.Parameter(gumbel_softmax(self.w_theta, u_w, self.temp, hard=True), requires_grad=False)
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        if self.testing == 0:
            self.w = self.w_z * (self.w_mu + sigma_w * epsilon_w)
        else:
            self.w = self.w_z_extra * (self.w_mu + sigma_w * epsilon_w)

        if self.v_mu is not None:
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            if self.freeze == 0:
                u_v = torch.zeros_like(self.v_mu).uniform_(0.0, 1.0)
                self.v_z = gumbel_softmax(self.v_theta, u_v, self.temp, hard=True)
                self.v_z_extra = nn.Parameter(gumbel_softmax(self.v_theta, u_v, self.temp, hard=True), requires_grad=False)
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            if self.testing == 0:
                self.v = self.v_z * (self.v_mu + sigma_v * epsilon_v)
            else:
                self.v = self.v_z_extra * (self.v_mu + sigma_v * epsilon_v)

        else:
            self.v = None

        if self.training:
            w_gamma = sigmoid(self.w_theta)

            kl_w_gamma = w_gamma * (torch.log(w_gamma) - torch.log(self.gamma_prior)) + \
                    (1 - w_gamma) * (torch.log(1 - w_gamma) - torch.log(1 - self.gamma_prior)) 

            kl_w = w_gamma *(torch.log(self.sigma_0) - torch.log(sigma_w) +
                    0.5*(sigma_w ** 2 + self.w_mu ** 2)/self.sigma_0**2 - 0.5)

            if self.v_mu is not None:
                v_gamma = sigmoid(self.v_theta)        

                kl_v_gamma = v_gamma * (torch.log(v_gamma) - torch.log(self.gamma_prior)) + \
                            (1 - v_gamma) * (torch.log(1 - v_gamma) - torch.log(1 - self.gamma_prior))        
        
                kl_v = v_gamma *(torch.log(self.sigma_0) - torch.log(sigma_v) +
                        0.5*(sigma_v ** 2 + self.v_mu ** 2)/self.sigma_0**2 - 0.5)

                self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_v_gamma) + torch.sum(kl_w) + torch.sum(kl_v)

            else:
                self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_w)
        
        return F.linear(input, self.w.T, self.v)

##########################################################################################################################################
#### Gaussian without spike-and-slab linear layer
class Gauss_layer(nn.Module):
    
    __constants__ = ['bias', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, bias=True, sigma_0=1):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(output_dim))
            self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
    
    def extra_repr(self):
        return 'input_dim={}, output_dim={}, bias={}'.format(
            self.input_dim, self.output_dim, self.v_mu is not None
        )

    def forward(self, input):
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        self.w = self.w_mu + sigma_w * epsilon_w

        if self.v_mu is not None:
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            self.v = self.v_mu + sigma_v * epsilon_v
        else:
            self.v = None

        if self.training:
            kl_w = (torch.log(self.sigma_0) - torch.log(sigma_w) +
                        0.5*(sigma_w ** 2 + self.w_mu ** 2)/self.sigma_0**2 - 0.5)
        
            if self.v_mu is not None:
                kl_v = (torch.log(self.sigma_0) - torch.log(sigma_v) +
                        0.5*(sigma_v ** 2 + self.v_mu ** 2)/self.sigma_0**2 - 0.5)
                
                self.kl = torch.sum(kl_w) + torch.sum(kl_v)  
            else:
                self.kl = torch.sum(kl_w)

        return F.linear(input, self.w.T, self.v)