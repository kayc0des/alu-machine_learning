# Project: Building a Binary Image Classifier from Scratch using NumPy

## Inroduction
Supervised learning is a type of machine learning where the model learns from labeled data, meaning it is provided with input-output pairs during training. The goal of supervised learning is to learn a mapping from inputs to outputs, so that given new, unseen inputs, the model can accurately predict the corresponding outputs. Supervised learning can be divided into two main categories: classification and regression. For this project we will focus on classification.

- Classification: Classification is a type of supervised learning task where the goal is to predict the category or class label of a given input based on its features. In classification, the output variable is categorical, meaning it takes on discrete values from a predefined set of classes. For example, recognizing handwritten digits as numbers from 0 to 9, in this project we will build a binary image classifier to recognize handwritten digits - 0 and 1.

## Data Description
This project utilizes the following datasets stored in the `data` directory:

- [`Binary_Train.npz`](data/Binary_Train.npz): Dataset containing training data for binary image classification.
- [`Binary_Dev.npz`](data/Binary_Dev.npz): Dataset containing development (validation) data for binary image classification.
- [`MNIST.npz`](data/MNIST.npz): Dataset containing handwritten digit images for multiclass image classification.

## Single Neuron Training (Project 0 - 7)
This project focuses on training a single neuron for binary image classification. The `Neuron` class is implemented to perform forward propagation, cost calculation, gradient descent, and training using a specified number of iterations and learning rate.

### Training Results
The training process includes monitoring the cost function over iterations and plotting the training cost graphically. Below are the visualizations generated during training and some key stats.

- Train cost: 0.013386353289868338
- Train accuracy: 99.66837741808132%
- Dev cost: 0.010803484515167203
- Dev accuracy: 99.81087470449172%

#### Training Cost
![Training Cost](img/Neuron_training_cost.png)

#### Predicted Output
![Predicted Output](img/predicted_output.png)

## Learning Objectives
At the end of this project, you'll be proficient in:

- Understanding the concept of a model and its role in machine learning.
- Grasping the fundamentals of supervised learning and its significance in training models.
- Defining prediction and its application in making estimations based on trained models.
- Recognizing nodes as the basic computational units in neural networks.
- Understanding the significance of weights and biases in adjusting model behavior.
- Explaining activation functions and their role in introducing non-linearity into neural networks.
- Understanding and implementing Sigmoid, Tanh, ReLU, and Softmax activation functions.
- Understanding the concept of layers in neural networks and their role in hierarchical feature learning.
- Identifying hidden layers as layers within a neural network that are neither input nor output layers.
- Understanding Logistic Regression as a linear classifier used for binary classification tasks.
- Defining loss and cost functions and understanding their role in evaluating model performance.
- Understanding forward propagation as the process of predicting outputs given inputs in a neural network.
- Understanding Gradient Descent as an optimization algorithm used for minimizing loss functions.
- Understanding backpropagation as the process of computing gradients of the loss function with respect to weights and biases.
- Understanding Computation Graphs as a visual representation of mathematical operations in neural networks.
- Understanding the importance of initializing weights and biases appropriately in neural networks.
- Understanding the importance of vectorization in efficiently processing large datasets.
- Understanding how to split data into training, validation, and test sets for model evaluation.
- Understanding multiclass classification and its differences from binary classification.
- Understanding one-hot encoding as a technique used to represent categorical variables.
- Understanding softmax function and its application in multiclass classification tasks.
- Understanding cross-entropy loss as a loss function commonly used in classification tasks.
- Understanding pickling in Python and its role in serializing and deserializing objects.