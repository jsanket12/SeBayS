#!/bin/bash
#SBATCH --job-name=BayS     # create a short name for your job
#SBATCH --account=csigeneral           # csigeneral-v mlg-core oddr
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

##SBATCH --mail-type=BEGIN        # send email when job begins
##SBATCH --mail-type=END          # send email when job ends
##SBATCH --mail-user=sjantre@bnl.gov

#SBATCH --output=run_BayS_WRN_28_10_C100_%j.log

source ~/.bashrc
conda activate torch

# # ===================================================================================================================================
# # Dense BNN Ensemble
# # ===================================================================================================================================
# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 3 --seed 1 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 0.2

# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 3 --seed 2 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 0.2

# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 3 --seed 3 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 0.2

# ====================================================================

# python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 3 --seed 1 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005  \
#                     --prior_mean 0.0 --prior_std 1.0

# python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 3 --seed 2 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005  \
#                     --prior_mean 0.0 --prior_std 1.0

# python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 3 --seed 3 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.0 \
#                     --epochs 250 --density 1 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005  \
#                     --prior_mean 0.0 --prior_std 1.0

# # ===================================================================================================================================
# # Bayesian Subnetwork Ensemble
# # ===================================================================================================================================
# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 1 --seed 1 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
#                     --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 1.0

# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 1 --seed 2 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
#                     --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 1.0

# python main_BayS.py --model wrn-28-10 --data cifar10 --ens_seed 1 --seed 3 --kl_anneal 150 --lr_std_diff --steps_diff \
#                     --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
#                     --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                     --prior_mean 0.0 --prior_std 1.0

# ====================================================================

python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 1 --seed 1 --kl_anneal 150 --lr_std_diff --steps_diff \
                    --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
                    --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
                    --prior_mean 0.0 --prior_std 0.2

python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 1 --seed 2 --kl_anneal 150 --lr_std_diff --steps_diff \
                    --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
                    --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
                    --prior_mean 0.0 --prior_std 0.2

python main_BayS.py --model wrn-28-10 --data cifar100 --ens_seed 1 --seed 3 --kl_anneal 150 --lr_std_diff --steps_diff \
                    --update-frequency 1000 --batch-size 128 --death-rate 0.5 --sparsity_distribution ERK \
                    --epochs 250 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
                    --prior_mean 0.0 --prior_std 0.2

# # ===================================================================================================================================
# # Sequential Bayesian Subnetwork Ensemble
# # ===================================================================================================================================
# python main_SeBayS.py --model wrn-28-10 --data cifar10 --ens_seed 1 --epochs-explo 150 --model-num 3 --kl_anneal 150 --lr_std_diff \
#                       --update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 --sparsity_distribution ERK \
#                       --epochs 450 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                       --prior_mean 0.0 --prior_std 0.2

# python main_SeBayS.py --model wrn-28-10 --data cifar100 --ens_seed 1 --epochs-explo 150 --model-num 3 --kl_anneal 150 --lr_std_diff \
#                       --update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 --sparsity_distribution ERK \
#                       --epochs 450 --density 0.5 --valid_split 0.1 --use_bnn --init_lr 0.1 --weight_decay 0.0005 \
#                       --prior_mean 0.0 --prior_std 0.2

conda deactivate