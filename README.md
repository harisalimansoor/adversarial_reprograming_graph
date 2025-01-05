# Adversarial Reprogramming on Graph Data
## Overview
This project extends the concept of adversarial reprogramming for graph data, applying it to the MNIST dataset. The goal is to demonstrate how adversarial reprogramming can be utilized to attack graph neural networks. The concept of reprogramming machine learning models without direct access to their internal weights or gradients is an interesting domain for research, as shown in the paper:

Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with Scarce Data and Limited Resources.

This project uses adversarial techniques to adjust the weights of a model to solve a modified version of the MNIST dataset, which is used as a multi-label classification task.

## Dependencies
To run this code, you will need to install the following libraries:

Python 3.6+
TensorFlow 2.x
Keras
NumPy
scikit-learn
Matplotlib


## Files
main.py
This is the main script for the project. It contains the implementation of:

Data preparation: Loads and processes the MNIST dataset, reshapes it, and splits it into training, validation, and test sets.
Model design: A simple neural network model is created for training on the MNIST dataset.
Loss function: Implements the multi-label loss function that will be used to train the model.
Adversarial reprogramming: Implements the adversarial reprogramming approach to train a base MNIST model and map it to a new output space with only a subset of labels.
Training: The model is trained using gradient descent with adversarial updates.
results.txt
This file logs the results of the training process, including the loss values and model metrics (accuracy, etc.) for each epoch.

model_minist_epoch_1.h5
This file is a saved model after training the MNIST dataset. It is used for testing and adversarial reprogramming.

results1.txt
Logs the configuration details like batch size, q (number of times gradient is computed), and beta (a parameter used for zero-order optimization).

## Running the Code
To train the model, simply run the main.py script:

python main.py
This will begin the training process using adversarial reprogramming with the MNIST dataset.

## Configuration
In the script, you can adjust the following parameters:

Batch size: Set the batch_size variable to define the batch size for training.
Number of epochs: Modify the epochs variable to set how many epochs the model will train.
Model path: Specify the path where the trained model will be saved in the saved_model_name variable.
Adversarial reprogramming settings: Adjust the multi-label mapping, zero-order optimization (q and beta), and layer selection.
Logging and Results
The results will be logged in the results.txt and results1.txt files. These logs include the training loss, validation loss, and other metrics. After the training, the best model will be saved to the specified saved_model_name.

## Research Motivation
The goal of this project is to explore the application of adversarial reprogramming on graph data, using a simple dataset (MNIST) to demonstrate the feasibility of the technique. The concept is based on the idea that we can manipulate a trained model to perform tasks on a new set of data without retraining the model entirely.

## References
Tsai, Y.-Y., Chen, P.-Y., & Ho, T.-Y. (2018). Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with Scarce Data and Limited Resources.

