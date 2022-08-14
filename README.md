# swin_transformer_tensorflow
The repository provides an implementation of swin transformer for image classification on CIFAR-100 dataset.  
The implementation is based on the paper (Swin Transformer: Hierarchical Vision Transformer using Shifted Windows)[https://arxiv.org/abs/2103.14030].  

The Swin Transformer (Shifted WINdow Transformer) computes the hierarchical representations using shifted windows that limits the self attention computation to non-overlapping windows while simultaneously allowing for cross-window connections.  

The implementation requires Tensorflow and TensorFlow Addons so make sure to install them before running the code. 

The implemented architecture is quite limited in terms of parameters, i.e. 152K and is trained with 50 epochs only. You can vary the hyperparameters and implement a different backbone in order to improve the recognition results. For the current implementation, the train-validation loss graph is shown below.

![res1](https://user-images.githubusercontent.com/26203136/184553011-9510bc07-06b8-4b48-8771-33e322a7cbad.png)


