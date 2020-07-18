# Handwritten Character Recognition using Deep Learning

Basic CNN model trained using [MNIST](http://yann.lecun.com/exdb/mnist/) and [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset to predict handwritten characters (letters and digits), each image is resized to grayscale 28x28px image. The model included in this repo is not perfect as I'm still learning to improve it. More explanation below:

### The Data

NIST characters dataset    
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/images/nist.png) 

MNIST digits dataset  
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/images/mnist.png)

To add more relevant data, ***Data Augmentation*** with the following properties were added:
   * Rotation (10 degree)
   * Scaling (10%)
   * Shifting (10%)


### CNN Architecture
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/images/architecture.png)

Then the model is trained using the ADAM gradient descent algorithm, logarithmic loss, and a mini-batch gradient descent with mini-batch size 64 then saved model will be used to predict canvas image in Tkinter GUI.

App Demo  
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/demo.gif)  

### Prerequisites

* Python 3.5 and up
* Tkinter
* Tensorflow
* Keras
* Scikit-learn
* Pillow 7.1.2

### Running
1. Put model and model in the same folder as main.py
1. Run main.py

## Built With

* [Python 3.8.1](https://www.python.org/) - The main programming language used
* [Tensorflow 2.2.0](https://www.tensorflow.org/) - One of the best ML library to ease developing
* [Tkinter 8.6](https://tkdocs.com/) - Used to make the program GUI
* [Pycharm 2020.1.2](https://www.jetbrains.com/pycharm/) - Main Python IDE used
