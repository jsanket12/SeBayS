#!/bin/bash
#SBATCH --job-name=dense_BNN_Ab_WRN_C10      # create a short name for your job
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

#SBATCH --output=WRN_28_10_C10_dense_BNN_Ablation_Results_%j.log

source ~/.bashrc
conda activate torch

# resume=./results/wrn-28-10/cifar10/wd_0.0005/random/seed_5/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# resume=./results/wrn-28-10/cifar10/wd_0.0005/death_0.8_large_death_0.8/random/seed_1/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# resume=./results/wrn-28-10/cifar10/wd_0.0005/random/seed_5/density_0.1/SeBayS/ERK/sig_0.2/anneal_0/lr_std_diff/\M=7/
# resume=./results/wrn-28-10/cifar10/wd_0.0005/ens_seed_3/density_0.2/BayS/ERK/sig_1.0/anneal_0/steps_diff/lr_std_diff/
# resume=./results/wrn-28-10/cifar10/wd_0.0005/ens_seed_1/density_1.0/BayS/Uniform/sig_0.2/anneal_150/lr_std_diff/
# python ensemble_learners.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 1 --use_bnn
# python ensemble_learners.py --mode disagreement --resume $resume --dataset cifar10 --model wrn-28-10 --seed 1 --use_bnn
# python ensemble_learners.py --mode KD --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 1 --test-batch-size 500 --corrupt --use_bnn

# python ensemble_learners.py --mode ood --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --test-batch-size 200 --ood-dataset cifar100 --use_bnn
# python ensemble_learners.py --mode ood --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --test-batch-size 200 --ood-dataset svhn --use_bnn
# python ensemble_learners.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --epsilon 0.03137 --test-batch-size 200 --use_bnn
# # python ensemble_learners.py --mode tsne --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --test-batch-size 1000 --use_bnn

# # resume=./results/wrn-28-10/cifar10/wd_0.0005/random/seed17/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# python ensemble_learners.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --epsilon 0.03137 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar10/wd_0.0005/random/seed17/density_0.1/SeBayS/ERK/sig_0.2/anneal_0/lr_std_diff/\M=7/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --epsilon 0.03137 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar10/wd_0.0005/density_0.2/BayS/ERK/sig_1.0/anneal_0/steps_diff/lr_std_diff/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --epsilon 0.03137 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar10/wd_0.0005/density_1.0/BayS/Uniform/sig_1.0/anneal_150/steps_diff/lr_std_diff/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --epsilon 0.03137 --test-batch-size 200 --use_bnn

# python ensemble_learners_MC.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --use_bnn
# python ensemble_learners_MC.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --use_bnn
# python ensemble_learners_MC.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --test-batch-size 500 --corrupt --use_bnn
# python ensemble_learners_MC.py --mode disagreement --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --use_bnn
# python ensemble_learners_MC.py --mode KD --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --use_bnn
# python ensemble_learners_MC.py --mode adv --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --epsilon 0.03137254901 --test-batch-size 1 --use_bnn
# python ensemble_learners_MC.py --mode ood --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --test-batch-size 500 --ooddataset cifar100 --use_bnn
# python ensemble_learners_MC.py --mode ood --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --num-MC-Test 3 --test-batch-size 500 --ooddataset svhn --use_bnn

# # resume=./results/wrn-28-10/cifar100/wd_0.0005/random/seed_5/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# # resume=./results/wrn-28-10/cifar100/wd_0.0005/death_0.8_large_death_0.8/random/seed_1/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# # resume=./results/wrn-28-10/cifar100/wd_0.0005/random/seed_5/density_0.1/SeBayS/ERK/sig_0.2/anneal_0/lr_std_diff/\M=7/
# # resume=./results/wrn-28-10/cifar100/wd_0.0005/death_0.5_large_death_0.8/ens_seed_3/density_0.2/BayS/ERK/sig_0.2/anneal_0/steps_diff/lr_std_diff/
# resume=./results/wrn-28-10/cifar100/wd_0.0005/death_0.5_large_death_0.8/ens_seed_3/density_1.0/BayS/Uniform/sig_1.0/anneal_0/steps_diff/lr_std_diff/

# resume=./results/wrn-28-10/cifar100/wd_0.0005/ens_seed_1/density_1.0/BayS/Uniform/sig_0.2/anneal_150/lr_std_diff/
# python ensemble_learners.py --mode predict --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode disagreement --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode KD --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --use_bnn
# python ensemble_learners.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 500 --corrupt --use_bnn

# python ensemble_learners.py --mode ood --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 200 --ood-dataset cifar10 --use_bnn
# python ensemble_learners.py --mode ood --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 200 --ood-dataset svhn --use_bnn
# python ensemble_learners.py --mode adv --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --epsilon 0.03137254901 --test-batch-size 200 --use_bnn
# python ensemble_learners.py --mode tsne --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 1000 --use_bnn

# # seed=1
# # resume=./results/wrn-28-10/cifar100/wd_0.0005/random/seed_$seed/density_0.2/SeBayS/ERK/sig_0.2/anneal_150/lr_std_diff/\M=3/
# python ensemble_learners.py --mode adv --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --epsilon 0.03137254901 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar100/wd_0.0005/random/seed17/density_0.1/SeBayS/ERK/sig_0.2/anneal_0/lr_std_diff/\M=7/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --epsilon 0.03137254901 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar100/wd_0.0005/density_0.2/BayS/ERK/sig_0.2/anneal_0/steps_diff/lr_std_diff/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --epsilon 0.03137254901 --test-batch-size 200 --use_bnn

# resume=./results/wrn-28-10/cifar100/wd_0.0005/density_1.0/BayS/Uniform/sig_1.0/anneal_0/steps_diff/lr_std_diff/
# python ensemble_learners_new.py --mode adv --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --epsilon 0.03137254901 --test-batch-size 200 --use_bnn

conda deactivate