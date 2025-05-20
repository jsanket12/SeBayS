import torch
import torchvision
import torch.nn.functional as F


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )


def get_weighted_layers(model, i=0, layers=None, layer_names=None, linear_layers_mask=None, use_bnn=False):
    if layers is None:
        layers = []
    if layer_names is None:
        layer_names = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            layer_names.append(layer_name)
            linear_layers_mask.append(1)
            if use_bnn:
                linear_layers_mask.append(1)
        elif hasattr(p, 'weight') and type(p) not in EXCLUDED_TYPES:
            layers.append([p])
            layer_names.append(layer_name)
            linear_layers_mask.append(0)
            if use_bnn:
                linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(p, torchvision.models.resnet.BasicBlock):
            _, _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, layer_names=layer_names, linear_layers_mask=linear_layers_mask, use_bnn=use_bnn)
        else:
            _, _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, layer_names=layer_names, linear_layers_mask=linear_layers_mask, use_bnn=use_bnn)

    return layers, layer_names, linear_layers_mask, i 



def get_W(model, return_linear_layers_mask=False, use_bnn=False):
    layers, _, linear_layers_mask, _ = get_weighted_layers(model, use_bnn=use_bnn)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)
        if use_bnn:
            W.append(layer[idx].posterior_std)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W

def get_W_ERK(model, use_bnn=False):
    layers, layer_names, _ , _ = get_weighted_layers(model)

    W_mean = []
    W_std = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W_mean.append(layer[idx].weight)
        if use_bnn:
            W_std.append(layer[idx].posterior_std)

    return layers, layer_names, W_mean, W_std


def get_grad(model, data, target, forward_with_mean, use_bnn=True):
    if forward_with_mean:
        model.set_test_mean(True)
    else:
        model.set_test_mean(False)
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return get_W(model, use_bnn=use_bnn)