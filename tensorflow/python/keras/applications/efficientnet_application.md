EfficientNet for Transfer Learning

## Introduction to EfficientNet
EfficientNet, first introduced in https://arxiv.org/abs/1905.11946 is among the most efficient models (i.e. requiring least FLOPS for inference) that reaches SOTA in both imagenet and common image classification transfer learning tasks. 

The base model is similar to MnasNet (https://arxiv.org/abs/1807.11626), where the core goal is reaching near-SOTA with a significantly faster model. By introducing a heuristic way to scale the model, EfficientNet provides a family of models (B0 to B7) that represents a good combination of efficiency and accuracy on a variety of scales. Such a scaling heuristics (compound-scaling, details see https://arxiv.org/abs/1905.11946) allows the efficiency-oriented base model (B0) to surpass models at every scale, while avoiding extensive grid-search of hyperparameters. 

A summary of the latest updates on the model is available at https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet, where various augmentation schemes and semi-supervised learning approaches are applied to further improve the imagenet performance of the models. These extensions of the model can be used by updating weights but not changing model architecture. Hence the EfficientNet architecture is a growing family of SOTA models for image classification tasks at various scales.

## Compound scaling

The EfficientNet models are approximately created using compound scaling. Starting from the base model B0, as model size scales from B0 to B7, the extra computational resource is proportioned into width, depth and resolution of the model by requiring each of the three dimensions to grow at the same power of a set of fixed ratios. 

However, it must be noted that the ratios are not taken accurately. A few points need to be taken into account:
Resolution. Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries of some layers which wastes computational resources. This especially applies to smaller variants of the model, hence the input resolution for B0 and B1 are chosen as 224 and 240. 
Depth and width. Channel size is always rounded to 8/16/32 because of the architecture. 
Resource limit. Perfect compound scaling would assume spatial (memory) and time allowance for the computation to grow simultaneously. However OOM may further bottleneck the scaling of resolution. 

As a result, compound scaling factor is significantly off from https://arxiv.org/abs/1905.11946. Hence it is important to understand the compound scaling as a rule of thumb that leads to this family of base models, rather than an exact optimization scheme. This also justifies that in the keras implementation (detailed below), only these 8 models, B0 to B7, are exposed to the user and arbitrary width / depth / resolution is not allowed. 

## Keras implementation of EfficientNet

An implementation of EfficientNet B0 to B7 has been shipped with tf.keras since TF2.3. To use EfficientNetB0 for classifying 1000 classes of images from imagenet, run:
```
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights=’imagenet’)
```
This model takes input images of shape (224, 224, 3), and the input data should range [0,255]. Resizing and normalization are included as part of the model.

Because training EfficientNet on imagenet takes a tremendous amount of resources and several techniques that are not a part of the model architecture itself. Hence the Keras implementation by default loads pre-trained weights with AutoAugment (https://arxiv.org/abs/1805.09501). 

For B0 to B7 base models, the input shapes are different. Here is a list of input shape expected for each model:
EfficientNetB0: 224,
EfficientNetB1: 240,
EfficientNetB2: 260,
EfficientNetB3: 300,
EfficientNetB4: 380,
EfficientNetB5: 456,
EfficientNetB6: 528,
EfficientNetB7: 600

When the use of the model is intended for transfer learning, the Keras implementation provides a option to remove the top layers:
```
model = EfficientNetB0(include_top=False, weights=’imagenet’)
```
This option excludes the final Dense layer that turns 1280 features on the penultimate layer into prediction of the 1000 classes in imagenet. Replacing the top with custom layers allows using EfficientNet as a feature extractor and transfers the pretrained weights to other tasks. 




## Example: Fine tuning EfficientNetB0 for CIFAR-100.

As an architecture, EfficientNet is capable of a wide range of image classification tasks. For example, using the CIFAR-100 dataset, training the model takes only 20 seconds per epoch on TPU (v2 that is available on colab). This might make it sounds easy to simply train EfficientNet on any dataset wanted from scratch. 

However, training EfficientNet on smaller datasets, especially those with lower resolution like CIFAR-100, faces the significant challenge of overfitting or getting trapped in local extrema. Using EfficientNetB0 to train this dataset may not be extremely expensive in resource, but requires very careful choice of hyperparameters and is difficult to find suitable regularization. Instead, using pre-trained imagenet weights and only transfer learn (fine-tune) the model allows making use of the power of EfficientNet much easier. Below is an example of training EfficientNetB0 on CIFAR-100 from scratch. 




The first step to transfer learning is to freeze all layers and train only the top layers. For this step a relatively large learning rate (~0.1) can be used to start with, while applying some learning rate decay (either ExponentialDecay or use ReduceLRPlateau callback). On CIFAR-100 with EfficientNetB0, this step will take validation accuracy to ~70% with suitable (but not absolutely optimal) image augmentation. For this stage, using EfficientNetB0, validation accuracy and loss will be consistently better than training accuracy and loss. This is because the regularization is relatively strong, and it only suppresses train time metrics. 
[figure]

The second step is to unfreeze a number of layers. Unfreezing all layers and fine tuning is usually thought to only provide incremental improvements on validation accuracy, but for the case of EfficientNetB0 it boosts validation accuracy to ~84%, while reaching ~87% as in the original paper requires more advanced augmentation. 

Two important points on unfreezing layers:
The batch normalization layers need to be kept untrainable (https://keras.io/guides/transfer_learning/). If they are also turned to trainable, the first epoch after unfreezing will significantly reduce accuracy.
Each block needs to be all turned on or off. This is because the architecture includes a shortcut from the first layer to the last layer for each block. Not respecting blocks also significantly harms the final performance.



Larger variants of EfficientNet do not guarantee improved performance, especially for tasks with less data or fewer classes. In such a case, the larger variant of EfficientNet chosen, the harder it is to tune hyperparameters. 
EMA is extremely important in training EfficientNet from scratch, but not so much for transfer learning.
Do not use the RMSprop setup as in the original paper for transfer learning, at least not at first. The momentum and learning rate are too high for transfer learning. It will easily corrupt the pretrained weight and blow up the weight. A quick check is to see if loss (as categorical cross entropy) is getting significantly larger than log(NUM_CLASSES) after the same epoch. If so, the initial learning rate/momentum is too high.


## Using the latest EfficientNet weights

Since the initial paper, the EfficientNet has been improved by various methods for data preprocessing and for using unlabelled data to enhance learning results. These improvements are relatively hard and computationally costly to reproduce, and require extra code; but the weights are readily available in the form of TF checkpoint files. The model architecture has not changed, so loading the improved checkpoints is possible.

To use a checkpoint provided at (CITE), first download the checkpoint. As example, here we download noisy-student version of B1
```
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b1.tar.gz
tar -xf noisy_student_efficientnet-b1.tar.gz
```

Then use the script efficientnet_weight_update_util.py to convert ckpt file to h5 file. 
```
python efficientnet_weight_update_util.py --model b1 --notop --ckpt efficientnet-b1/model.ckpt --o efficientnetb0_notop.h5
```

When creating model, use
```
model = EfficientNetB0(weights=’efficientnetb0_notop.h5’, include_top=False)
```
To load the new weights.



Confirmed: it does converge considerably faster than from scratch. 
Now trying: different augs (getting to 73%).
Next up: warm up training; nestorov; Trying to recover 80% -> see what keras generator does

Experimenting

What work and what not:
Smaller batch size is good for small dataset (64 better than 128), possibly because of regularizing.
Nesterov does not help
Warming up could make it faster, no performance improvement. 

A current working case reaching 84: warm up; 30 epochs freezing; __ epochs finetuning 0.001 lr to start.  Shear __, rot __, shift __.


Scrambling figure


When training is significantly outperforming validation it means overfitting is getting severe. For the case of CIFAR100, using EfficientNetB0 does not have much of overfitting problem for only tuning the top layers; but when opening up all layers for fine tuning, while improving performance, the overfitting effect start to bottleneck the training as training loss drop very fast. 
One of the approaches to cope with the situation is to reduce learning rate for the fine tuning phase as suggested in (Cite francois guide), or use other existing regularizing methods; while another [[trying]] idea is to make a phased reopening.  

