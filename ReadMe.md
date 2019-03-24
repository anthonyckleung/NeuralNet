## Simple Neural Network on the Iris Data set

### Description
This is a small vanilla neural network I created for the Iris data set.
The program allows a variable number of layers and nodes/neurons per layer.
The architecture of the program was inspired from the ["Neural Networks and Deep
Learning"](http://neuralnetworksanddeeplearning.com/about.html) ebook by Michael Nielsen. 

### Prerequsities
- numpy
- sklearn
- matplotlib

To run the program without any hidden layers or any adjustments on hyperparameters, simply execute(*) :

`python neuralnet.py`



Another way is with flags. I have implemented a few flags that allows one to adjust
hyperparameters of the neural network.

| Flag | Details |
|------|---------|
|`-hl` | Hidden layer configurations (Default: N/A)|
|`-lr` | Learning rate (Default: 0.05)|
|`-m` | Momentum (Default: 0.05)|
|`-rg` | L_2 regularization (lambda) (Default: 0.001)|
|`-ep` | Total number of epochs (Default: 1000)|

For example, to run the neural network with 2 sequential hidden layers with the
first having 4 nodes and the next having 6 nodes, the command is:

`python neuralnet.py -hl 4 6`


-------
(*) This assumes one has Python3 has the only interpreter. Otherwise one would run with `python3`.
