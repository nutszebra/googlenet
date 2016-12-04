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
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [256, 512], then 224x224 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 384x384, then they are normalized locally. Single image test is used to calculate total accuracy.  

# Cifar10 result

| network              | depth  | total accuracy (%) |
|:---------------------|--------|-------------------:|
| my implementation    | 22     | soon               |

<img src="https://github.com/nutszebra/googlenet/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/googlenet/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Going Deeper with Convolutions [[1]][Paper]

[paper]: https://arxiv.org/abs/1409.4842 "Paper"
