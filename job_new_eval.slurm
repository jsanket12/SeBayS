#!/bin/bash
#SBATCH --job-name=MCD_WRN_C100      # create a short name for your job
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

#SBATCH --output=WRN_28_10_C100_MCD_Results_%j.log

source ~/.bashrc
conda activate torch

# resume=./results_SnapShot/wrn-28-10/cifar10/seed_1/init_lr_0.2/cycle_len_50/C=3/
# python ensemble_learners_SnapShot.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode disagreement --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode KD --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --test-batch-size 500 --corrupt

# resume=./results_MCD/wrn-28-10/cifar10/wd_0.0005/batch_128/Drop_rate_0.1/
# python ensemble_learners_MCD.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18  --drop_rate 0.1 --mc_dropout
# python ensemble_learners_MCD.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
# python ensemble_learners_MCD.py --mode disagreement --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
# python ensemble_learners_MCD.py --mode KD --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
# python ensemble_learners_MCD.py --mode calibration --resume $resume --dataset cifar10 --model wrn-28-10 --seed 18 --test-batch-size 500 --corrupt --drop_rate 0.1 --mc_dropout

# resume=./results_SnapShot/wrn-28-10/cifar100/seed_1/init_lr_0.2/cycle_len_50/C=3/
# python ensemble_learners_SnapShot.py --mode predict --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode disagreement --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode KD --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18
# python ensemble_learners_SnapShot.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 500 --corrupt

resume=./results_MCD/wrn-28-10/cifar100/wd_0.0005/batch_128/Drop_rate_0.1/
python ensemble_learners_MCD.py --mode predict --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18  --drop_rate 0.1 --mc_dropout
python ensemble_learners_MCD.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
python ensemble_learners_MCD.py --mode disagreement --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
python ensemble_learners_MCD.py --mode KD --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --drop_rate 0.1 --mc_dropout
python ensemble_learners_MCD.py --mode calibration --resume $resume --dataset cifar100 --model wrn-28-10 --seed 18 --test-batch-size 500 --corrupt --drop_rate 0.1 --mc_dropout