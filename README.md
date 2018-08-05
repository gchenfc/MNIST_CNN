# MNIST_CNN
transparently coded Matlab implementation of a CNN for MNIST digit classification

## Purpose
Although Matlab already has a quite capable neural network and machine learning toolbox, I thought it prudent to hand-code a simple neural network as an exercise to get an idea of the underlying concepts and difficulties in NNs.

## Implementation
Currently, only a fully-connected layer with basic gradient descent for training is implemented.  It includes a sigmoid squashing function.  All intensive operations are done with matrix manipulations, as I tried my best to stay away from any higher-level matlab functions.

## Performance
TODO - measure
I think it was something like 92%
Matlab's was like 98%