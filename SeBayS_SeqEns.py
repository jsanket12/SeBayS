import argparse
import torch
torch.cuda.empty_cache()
import numpy as np
import time
import torch.nn as nn
import pandas as pd 
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
from IPython import display
# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
import os
import sys
sys.path.insert(1, '/lcrc/project/FastBayes/sanket_bnn/SeqEns_expts/')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from Gauss_resnet_models_new import SSGauss_Node_ResNet_Cifar
from Gauss_linear_layers import Gauss_layer
from Gauss_Conv_layers import SSGauss_Node_Conv2d_layer
from Gauss_BN_layers import Gauss_VB_BatchNorm2d
from data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders

parser = argparse.ArgumentParser(description='ResNet Cifar10')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--results_path', default='./results_resnet/', 
                        type = str, help = 'base path for saving result')

# Resnet Architecture
parser.add_argument('--depth', default=32, type=int, help='depth of the resnet')
parser.add_argument('--model-num', type=int, default=3, help='number of models in ensemble')

# Data setting
parser.add_argument('--data', type=str, default='cifar10')

# Training Setting
parser.add_argument('--total-epochs', type=int, default=450, help='total number of epochs to train ensemble')
parser.add_argument('--epochs-explo', type=int, default=150, help='training epochs of exploration phase')
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--perturb_factor', default = 3, type = int, help = 'perturbation factor to multiply stad dev with')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--valid_split', type=float, default=0.1)
parser.add_argument('--batch_size', default = 128, type = int, help = 'batch size for training')
# parser.add_argument('--test_batch_size', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
parser.add_argument('--freeze', action='store_true', default=False, help='Freeze sparsity after exploration. Default: True.')  

# Optimizer Setting
parser.add_argument('--clip', default = 1.0, type = float, help = 'Gradient clipping value')

# Prior Setting
parser.add_argument('--sigma_0', default = 0.2, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--temp', default = 0.5, type = float, help = 'temperature')
parser.add_argument('--gamma_prior', default = 0.0001, type = float, help = 'prior inclusion probaility for filters/nodes')

args = parser.parse_args()
print(args)

if args.freeze: 
    writer = SummaryWriter('runs/SS_Gaussian_Node_Freeze_RN_'+str(args.depth)+'_'+str(args.data)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor))
else:
    writer = SummaryWriter('runs/SS_Gaussian_Node_No_Freeze_RN_'+str(args.depth)+'_'+str(args.data)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor))

class RunManager():
    def __init__(self):
        # tracking every epoch count, loss, accuracy, time
        self.epoch_start_time = None
        self.run_data = []

    def begin_run(self):
        self.run_start_time = time.time()

    def begin_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self,epoch,train_loss,train_accuracy,
                    test_loss,test_accuracy,
                    edge_sparsity,param_pruned,flops_ratio,flops_pruned,
                    best_epoch,learning_rate,batch_size):
        # calculate epoch duration and run duration(accumulate)
        self.epoch_duration = time.time() - self.epoch_start_time
        self.run_duration = time.time() - self.run_start_time

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["epoch"] = epoch
        results["Train loss"] = train_loss    #loss
        results["Test loss"] = test_loss
        results["Train Accuracy"] = train_accuracy   #accuracy
        results["Test Accuracy"] = test_accuracy
        results["edge sparsity"] = edge_sparsity
        results["param pruned"] = param_pruned
        results["flops ratio"] = flops_ratio
        results["flops pruned"] = flops_pruned
        results["best_epoch"] = best_epoch
        results["epoch duration"] = self.epoch_duration
        results["run duration"] = self.run_duration
        results["lr"] = learning_rate
        results["batch_size"] = batch_size

        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        # display epoch information and show progress
        display.clear_output(wait=True)
        display.display(df)
  
    def save(self, Path, fileName):
        pd.DataFrame.from_dict(
            self.run_data, 
            orient = 'columns',
        ).to_csv(f'{Path+fileName}.csv')

def milestone_calculation(args):
    individual_epoch = (args.total_epochs - args.epochs_explo) / args.model_num
    args.individual_epoch = individual_epoch
    reset_lr_epochs1 = []
    epoch_ = args.epochs_explo
    for _ in range(args.model_num):
        reset_lr_epochs1.append(epoch_)
        epoch_ = epoch_ + individual_epoch
    reset_lr_epochs2 = np.array(reset_lr_epochs1) + individual_epoch / 2
    reset_lr_epochs1.pop(0)
    return np.ceil(reset_lr_epochs1), np.ceil(reset_lr_epochs2)

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data == 'cifar10':
        train_loader, valid_loader, _, _ = get_cifar10_dataloaders(args, args.valid_split,
                                                                max_threads=args.max_threads)
        target_dim = 10
    elif args.data == 'cifar100':
        train_loader, valid_loader, _, _ = get_cifar100_dataloaders(args, args.valid_split,
                                                                max_threads=args.max_threads)
        target_dim = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    m = RunManager()
    m.begin_run()

    sigma_0 = torch.as_tensor(args.sigma_0).to(device)
    temp = torch.as_tensor(args.temp).to(device)
    gamma_prior = torch.tensor(args.gamma_prior).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)

    net = SSGauss_Node_ResNet_Cifar(depth=args.depth, num_classes=target_dim, temp = temp, 
                            gamma_prior = gamma_prior, sigma_0 = sigma_0, testing = 0).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)
    milestone1, milestone2 = milestone_calculation(args)
    print(f'learning rate recycle minestone1 is {milestone1} minestone2 is {milestone2}')

    if args.freeze: 
        RESULTS_PATH = args.results_path + str(args.data) + '/ResNet' + str(args.depth) + \
                                        '/SS_Gauss_Node_SeqEns_Modified/Freeze/Perturb=%d/' \
                                        % (args.perturb_factor) + 'Seed=%d/' % (args.seed)
    else:
        RESULTS_PATH = args.results_path + str(args.data) + '/ResNet' + str(args.depth) + \
                                        '/SS_Gauss_Node_SeqEns_Modified/No_Freeze/Perturb=%d/' \
                                        % (args.perturb_factor) + 'Seed=%d/' % (args.seed)
    if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)

    MODELS_PATH = RESULTS_PATH + 'Models/'
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)

    ACTIVE_MODELS_PATH = RESULTS_PATH + 'Active_Models/'
    if not os.path.exists(ACTIVE_MODELS_PATH): os.makedirs(ACTIVE_MODELS_PATH)

    epoch = 1
    ens_iter = 0
    train_Loss = []
    train_Accuracy = []
    test_Loss = []
    test_Accuracy = []
    Edge_sparsity = []
    flops_ratio = []
    flops_pruned = []
    param_pruned = []

    # total_num_para = 0
    # for name, module in net.named_modules():
    #     if 'downsample' not in name:
    #         if isinstance(module, (SSGauss_Node_Conv2d_layer,Gauss_layer)):
    #             total_num_para += module.w_mu.numel()
    #             if module.v_mu is not None:
    #                 total_num_para += module.v_mu.numel()
    # print(total_num_para)

    # ResNet Version - 1
    if args.depth == 20:
        total_num_para = 268346 if target_dim == 10 else 274196
        total_flops = 40551050 if target_dim == 10 else 40556900
    elif args.depth == 32:
        total_num_para = 461882 if target_dim == 10 else 467732
        total_flops = 68862602 if target_dim == 10 else 68868452 
    elif args.depth == 44:
        total_num_para = 655418 if target_dim == 10 else 661268
        total_flops = 97174154 if target_dim == 10 else 97180004
    elif args.depth == 50:
        total_num_para = 752186 if target_dim == 10 else 758036
        total_flops = 111329930 if target_dim == 10 else 111335780
    elif args.depth == 56:
        total_num_para = 848954 if target_dim == 10 else 854804
        total_flops = 125485706 if target_dim == 10 else 125491556

    best_accuracy = 0
    best_epoch = 1
    NTrain = len(train_loader.dataset)
    NTest = len(valid_loader.dataset)

    if args.start_epoch > 1:
        if args.freeze: 
            RESULTS_PATH_RESUME = args.results_path + str(args.data) + '/ResNet' + str(args.depth) + \
                                        '/SS_Gauss_Node_SeqEns_Modified/Freeze/Perturb=3/Seed=%d/' % (args.seed)
        else:
            RESULTS_PATH_RESUME = args.results_path + str(args.data) + '/ResNet' + str(args.depth) + \
                                        '/SS_Gauss_Node_SeqEns_Modified/No_Freeze/Perturb=3/Seed=%d/' % (args.seed)
                                                
        # step_list = [idx for idx, element in enumerate(milestone1.tolist()) if element == args.start_epoch]
        # if step_list:
        dic = pd.read_csv(RESULTS_PATH_RESUME+'results_SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                    '_sig0_'+str(args.sigma_0)+'_perturb_3'+
                                    '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                    '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                    '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                    '_ens_models_'+ str(args.model_num)+'_Step_1.csv', index_col=0).to_dict('records')
        pre_load = []
        for i, sub in enumerate(dic, start = 0):
            if i < 150:
                pre_load.append(OrderedDict(sub))
        m.run_data = pre_load

        checkpoint = torch.load(RESULTS_PATH_RESUME+ 'Active_Models/' + 'model_'+str(args.start_epoch)+'.pt')
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(RESULTS_PATH_RESUME, checkpoint['epoch']))
        print(net.freeze)
        net.unfreeze_flag()
        print(net.freeze)
        if epoch == args.epochs_explo:
            optimizer.param_groups[0]['lr'] = 0.01
        elif epoch in milestone2:
            optimizer.param_groups[0]['lr'] = 0.001
        elif epoch in milestone1:
            optimizer.param_groups[0]['lr'] = 0.01
            print('\n------------ Perturbation Step: {}----------------'.format(ens_iter+1))
            trained_dict = net.state_dict()
            best_accuracy = 0
            
            for name, _ in net.named_parameters():
                if 'mu' in name:
                    temp_mu = trained_dict[name]
                if 'rho' in name:
                    temp_rho = trained_dict[name]
                    temp_sigma = torch.log(1 + torch.exp(temp_rho))
                    perturb = (torch.Tensor(temp_mu.shape).uniform_() > 0.5).float()
                    perturb[perturb==0] = -1
                    perturb = args.perturb_factor*perturb.to(device)*temp_sigma
                    trained_dict[name] = torch.Tensor(temp_sigma.shape).normal_(-6, .05)
                    name_mu = name.replace("rho", "mu")
                    trained_dict[name_mu] = temp_mu + perturb

            net.load_state_dict(trained_dict) 

            m.save(RESULTS_PATH,'results_SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                        '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                        '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                        '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                        '_ens_models_'+ str(args.model_num)+'_Step_'+str(ens_iter+1))
            ens_iter +=1
        epoch +=1

    while epoch <= args.total_epochs:
        print('----------Epoch {}----------------'.format(epoch))
        m.begin_epoch()
        net.train()
        train_loss = 0.
        correct_train = 0
        learning_rate = optimizer.param_groups[0]['lr']

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            output = net(images)
            nll_train = loss_func(output, labels)           
            kl_train = 0.
            for module in net.modules():
                if isinstance(module, (SSGauss_Node_Conv2d_layer,Gauss_VB_BatchNorm2d,Gauss_layer)):
                    kl_train += module.kl.div(NTrain)
            loss = nll_train + kl_train

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            train_loss += nll_train.mul(images.shape[0]).item()
            correct_train += output.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            net.eval()
            train_accuracy = correct_train / NTrain
            train_loss = train_loss/ NTrain

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)

            correct_test = 0
            test_loss = 0
            for _ , (images, labels) in enumerate(valid_loader):
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss = loss_func(output, labels)
                test_loss += loss.mul(images.shape[0]).item()
                correct_test += output.data.argmax(1).eq(labels.data).sum().item()

            test_accuracy = correct_test/ NTest
            test_loss = test_loss/ NTest

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch

            print('Best accuracy: {} and Epoch: {}'.format(best_accuracy,best_epoch))

            param_overall = 0.
            flops_overall = 0. 
            for name, module in net.named_modules():
                if 'downsample' not in name:
                    if isinstance(module, (SSGauss_Node_Conv2d_layer)): 
                        in_prune_channels = module.in_channels                   
                        out_prune_channels = (module.z != 0).sum()
                        if module.v is not None:
                            param_overall += (module.w != 0).sum() + (module.v != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                        else:
                            param_overall += (module.w != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                    elif isinstance(module, (Gauss_layer)):
                        if module.v is not None:
                            param_overall += (module.input_dim+1)*module.output_dim
                            flops_overall += (module.input_dim+1)*module.output_dim
                        else:
                            param_overall += module.input_dim*module.output_dim
                            flops_overall += module.input_dim*module.output_dim
            
            flops_pruned_val = flops_overall.item()
            param_pruned_val = param_overall.item()
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops

            flops_pruned.append(flops_pruned_val)
            param_pruned.append(param_pruned_val)
            flops_ratio.append(flops_ratio_val)
            Edge_sparsity.append(sparsity_overall)

            torch.save(net.state_dict(), MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                        '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                        '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                        '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                        '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt') 
       

        if epoch == args.epochs_explo or epoch in milestone2 or epoch in milestone1 or epoch==args.total_epochs:
            torch.save({
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, ACTIVE_MODELS_PATH + 'model_'+str(epoch)+'.pt')
                    
        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, \
                        Edge sparsity: {}, FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                            epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                            sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                        sparsity_overall,param_pruned_val,flops_ratio_val,flops_pruned_val,
                        best_epoch,learning_rate,args.batch_size)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)
        writer.add_scalar('data/duration_epoch', m.epoch_duration, epoch)
        writer.add_scalar('data/duration_run', m.run_duration, epoch)

        if args.freeze and args.start_epoch > 1 and net.freeze == 0:
            net.freeze_flag()
            print(net.freeze)
        if epoch == args.epochs_explo:
            optimizer.param_groups[0]['lr'] = 0.01
            if args.freeze:
                net.freeze_flag()
                print(net.freeze)
                for name, param in net.named_parameters():
                    if 'theta' in name: 
                        param.requires_grad = False

        elif epoch in milestone2:
            optimizer.param_groups[0]['lr'] = 0.001

        elif epoch in milestone1:
            optimizer.param_groups[0]['lr'] = 0.01
            print('\n------------ Perturbation Step: {}----------------'.format(ens_iter+1))
            trained_dict = net.state_dict()
            best_accuracy = 0

            for name, _ in net.named_parameters():
                if 'mu' in name:
                    temp_mu = trained_dict[name]
                if 'rho' in name:
                    temp_rho = trained_dict[name]
                    temp_sigma = torch.log(1 + torch.exp(temp_rho))
                    perturb = (torch.Tensor(temp_mu.shape).uniform_() > 0.5).float()
                    perturb[perturb==0] = -1
                    perturb = args.perturb_factor*perturb.to(device)*temp_sigma
                    trained_dict[name] = torch.Tensor(temp_sigma.shape).normal_(-6, .05)
                    name_mu = name.replace("rho", "mu")
                    trained_dict[name_mu] = temp_mu + perturb                

            net.load_state_dict(trained_dict)       
            
            m.save(RESULTS_PATH,'results_SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                        '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                        '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                        '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                        '_ens_models_'+ str(args.model_num)+'_Step_'+str(ens_iter+1))

            ens_iter +=1
        epoch +=1 

    print('Finished Training')
    writer.close()
    
    m.save(RESULTS_PATH,'results_SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                        '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                        '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                        '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                        '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                        '_ens_models_'+ str(args.model_num)+'_Step_'+str(ens_iter+1))

if __name__ == '__main__':
    main()