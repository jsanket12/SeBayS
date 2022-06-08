# SeBayS: Sequential Bayesian Neural Subnetwork Ensembles

[**Sanket Jantre**](https://jsanket123.github.io/), Sandeep Madireddy, Shrijita Bhattacharya, Tapabrata Maiti, Prasanna Balaprakash

**Abstract:** Deep neural network ensembles that appeal to model diversity have been used successfully to improve predictive performance and model robustness in several applications. Whereas, it has recently been shown that sparse subnetworks of dense models can match the performance of their dense counterparts and increase their robustness while effectively decreasing the model complexity. However, most ensembling techniques require multiple parallel and costly evaluations and have been proposed primarily with deterministic models, whereas sparsity induction has been mostly done through ad-hoc pruning. We propose sequential ensembling of dynamic Bayesian neural subnetworks that systematically reduce model complexity through sparsity-inducing priors and generate diverse ensembles in a single forward pass of the model. The ensembling strategy consists of an exploration phase that finds high-performing regions of the parameter space and multiple exploitation phases that effectively exploit the compactness of the sparse model to quickly converge to different minima in the energy landscape corresponding to high-performing subnetworks yielding diverse ensembles. We empirically demonstrate that our proposed approach surpasses the baselines of the dense frequentist and Bayesian ensemble models in prediction accuracy, uncertainty estimation, and out-of-distribution (OoD) robustness on CIFAR10, CIFAR100 datasets, and their out-of-distribution variants: CIFAR10-C, CIFAR100-C induced by corruptions. Furthermore, we found that our approach produced the most diverse ensembles compared to the approaches with a single forward pass and even compared to the approaches with multiple forward passes in some cases.

**Paper link:** [arXiv.2206.00794](https://arxiv.org/abs/2206.00794)

## How to Run Experiments (ResNet32-Cifar10)

### BNN sequential ensemble
**Training**
```
python BNN_SeqEns.py --data cifar10 --seed 1 --depth 32 --batch_size 128 \
  --total-epochs 450 --epochs-explo 150 --start-epoch 1 --model-num 3 --init_lr 0.1 --valid_split 0.1 \
  --sigma_0 0.2  --clip 1.0 --perturb_factor 3
```

**Evaluation**
```
results_path=results_resnet/cifar10/ResNet32/Gauss_SeqEns_Modified/
python BNN_SeqEns_Test.py --seed 1 --model-num 3 --perturb_factor 3 \
  --results_path $results_path --data cifar10 --total-epochs 450 --epochs-explo 150 --mode predict
```
    
### SeBayS-Freeze ensemble
**Training**
```
python SeBayS_SeEns.py --data cifar10 --seed 1 --depth 32 --batch_size 128 \
  --freeze --total-epochs 450 --epochs-explo 150 --start-epoch 1 --model-num 3 --init_lr 0.1 --valid_split 0.1 \
  --sigma_0 0.2 --clip 1.0 --perturb_factor 3
```
    
**Evaluation**
```
results_path=results_resnet/cifar10/ResNet32/SS_Gauss_Node_SeqEns_Modified/
python SeBayS_SeEns_Test.py --freeze --seed 1 --model-num 3 --perturb_factor 3 \
    --results_path $results_path --data cifar10 --total-epochs 450 --epochs-explo 150 --mode predict
```

### SeBayS-No Freeze ensemble
**Training**
```
python SeBayS_SeEns.py --data cifar10 --seed 1 --depth 32 --batch_size 128 \
  --total-epochs 450 --epochs-explo 150 --start-epoch 1 --model-num 3 --init_lr 0.1 --valid_split 0.1 \
  --sigma_0 0.2 --clip 1.0 --perturb_factor 3
```
    
**Evaluation**
```
results_path=results_resnet/cifar10/ResNet32/SS_Gauss_Node_SeqEns_Modified/
python SeBayS_SeEns_Test.py --seed 1 --model-num 3 --perturb_factor 3 \
    --results_path $results_path --data cifar10 --total-epochs 450 --epochs-explo 150 --mode predict
```
    
# Citation
If you find this repo helpful, please cite

```
@misc{SeBayS_2022,
  doi = {10.48550/arXiv.2206.00794},  
  url = {https://arxiv.org/abs/2206.00794},  
  author = {Jantre, Sanket and Madireddy, Sandeep and Bhattacharya, Shrijita and Maiti, Tapabrata and Balaprakash, Prasanna},
  title = {Sequential Bayesian Neural Subnetwork Ensembles},  
  year = {2022}
}
```
