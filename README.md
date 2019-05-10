# MNIST Classifier with Multilayered Perceptron
For those of us who don't know, MNIST is a huge dataset of 60k handwritten digits all labeled. The trick is to give a computer a new handwritten digit and see if it can identify the digit it was given.

[mnist-pic](https://en.wikipedia.org/wiki/MNIST_database#/media/File:MnistExamples.png)

# Why Perceptron

[rick-and-morty-meme](./img/rick-and-morty-meme.jpg)

The current state of the art says a CNN can learn this dataset with something like < 1% error. In fact, there is a tensorflow getting started tutorial where you build such a model. Using their models doesn't feel much like learning, but more like extra credit. 

[spongebob-meme](./img/spongebob-meme.jpg)

This is in part why I chose a perceptron. I know that I can manually implement such a model and see what's going on under the hood. I have followed the backpropogation example given by 3 blue 1 brown and built my model to have two hidden layers each with 16 neurons.

[model-image]()


