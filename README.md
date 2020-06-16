# Handwritten Character Recognition using Deep Learning

Basic CNN model trained using [MNIST](http://yann.lecun.com/exdb/mnist/) and [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset to predict handwritten characters (letters and digits), each image is resized to grayscale 28x28px image. The model included in this repo is not perfect as I'm still learning to improve it. More explanation below:

### The Data
NIST letters data, as you can see this data is imbalanced as letter I and F only has about 1k samples.  
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/NIST.PNG)  

MNIST digits train data  
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/mnist_train.PNG)

MNIST digits test data  
![](https://github.com/andreanlay/handwritten-character-recognition-deep-learning/blob/master/mnist_test.PNG)

At first, I tried to put all of the data without choosing specific number of samples, the trained model is called **model.h5** in the model folder, 
Because of NIST imbalanced data, there's a chance of overfitting so I try to get 1000 samples of each classes, eg. 1000 for A, 1000 for B, etc to train the model, this model is called **model_v2.h5**.
I saw a little improvement over the real-life testing when using V2 model, which is better ? You try!

### The Model
1. The first hidden layer is a 2D Convolution layer with the following properties:
    * 64 filters
    * Filter size = (5 x 5)
    * Input shape = (28 x 28 x 1)
    * ReLU activation function
2. The second hidden layer is a 2D Convolution layer with the following properties:
    * 32 filters
    * Filters size = (3 x 3)
    * ReLU activation function 
3. The next layer is a Max Pooling Layer with a pool size 2 x 2
4. After Max Pool layer, we have a Flatten layer that converts the matrix data to a vector to allows the output to be processed by standard Fully Connected (FC) layers
5. Next we have a fully connected layer with 36 hidden units with softmax activation function.

Then this model is trained using the ADAM gradient descent algorithm, logarithmic loss, and a mini-batch gradient descent with mini-batch size 512 then saved model will be used to predict canvas image in Tkinter GUI.

### Prerequisites

* Python 3.5 and up
* Tkinter
* Tensorflow
* Keras
* Scikit-learn
* Pillow 7.1.2

### Running
1. Put model and model_v2 in the same folder as main.py
1. Run main.py from CMD and you're ready to go

## Built With

* [Python 3.8.1](https://www.python.org/) - The main programming language used
* [Tensorflow 2.2.0](https://www.tensorflow.org/) - One of the best ML library to ease developing
* [Tkinter 8.6](https://tkdocs.com/) - Used to make the program GUI
* [Pycharm 2020.1.2](https://www.jetbrains.com/pycharm/) - Main Python IDE used
