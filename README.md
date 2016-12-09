# What's this
Implementation of GoogLeNet by chainer


# Dependencies

    git clone https://github.com/nutszebra/googlenet.git
    cd googlenet
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 


# Details about my implementation

* Data augmentation  
Train: Pictures are randomly resized in the range of [256, 512], then 224x224 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 384x384, then they are normalized locally. Single image test is used to calculate total accuracy. 

* Auxiliary classifiers  
No implementation

* Learning rate  
As [[1]][Paper] said, learning rate are multiplied by 0.96 at every 8 epochs. The description about initial learning rate can't be found in [[1]][Paper], so initial learning is setted as 0.0015 that is found in [[2]][Paper2].

* Weight decay  
The description about weight decay can't be found in [[1]][Paper], so by using [[2]][Paper2] and [[3]][Paper3] I guessed that weight decay is 2.0*10^-4.

# Cifar10 result

| network              | depth  | total accuracy (%) |
|:---------------------|--------|-------------------:|
| my implementation    | 22     | 91.33               |

<img src="https://github.com/nutszebra/googlenet/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/googlenet/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Going Deeper with Convolutions [[1]][Paper]  
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[2]][Paper2]  
Rethinking the Inception Architecture for Computer Vision [[3]][Paper3]  
[paper]: https://arxiv.org/abs/1409.4842 "Paper"
[paper2]: https://arxiv.org/abs/1502.03167 "Paper2"
[paper3]: https://arxiv.org/abs/1512.00567 "Paper3"
