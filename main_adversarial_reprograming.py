# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:53:12 2025

@author: haris_tuytv90
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:30:49 2023

@author: Haris


This is for a research project to extend adversarial reprograming on Graph data.

https://arxiv.org/pdf/1806.11146

Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with 
Scarce Data and Limited Resources, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

"""


import sys, os
file_path=os.path.dirname(os.path.abspath(__file__))
sys.path
sys.path.append(file_path)
os.chdir(file_path)

#%%

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




#%%
"""
create dataset, for this example we are using mnist dataset, there are total 60,000 samples, and 10 outputs
40,000 samples are used for training, 10,000 for validation and 10,000 for testing.

To train Multi-Label loss we used a subset of the MNIST dataset. the dataset of first three labels y=0,1,2 are used
there are 12430 samples and output is 0,1,2.  training:7450 testing:2486 valuation:2486 
"""

dataset_name = 'MNIST'
# Set batch size for training
batch_size = 4

# Load MNIST dataset using Keras
# X_train: training images, Y_train: training labels
# X_test: test images, Y_test: test labels
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Reshape images from 28x28 to 784 (flattening the 2D image to 1D array)
x_train = np.reshape(X_train, (-1, 784))
x_test = np.reshape(X_test, (-1, 784))

# Convert labels to one-hot encoded format
# e.g., 2 becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# Split the data into train, validation, and test sets
# Reserve last 10,000 samples for testing
x_test = x_train[-10000:]
y_test = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Reserve last 10,000 samples from remaining training data for validation
x_val = np.array(x_train[-10000:])
y_val = np.array(y_train[-10000:])
x_train = np.array(x_train[:-10000])
y_train = np.array(y_train[:-10000])

# Create TensorFlow datasets for full MNIST
# Training dataset with shuffling
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# Create subset of data containing only digits 0, 1, and 2
X_train_three_digits = []
Y_train_three_digits = []

# Filter data to keep only digits 0, 1, and 2
for i in range(len(x_train)):
    if (int(Y_train[i]) == 0) or (int(Y_train[i]) == 1) or (int(Y_train[i]) == 2):
        X_train_three_digits.append(x_train[i])
        Y_train_three_digits.append(Y_train[i])

# Convert labels to one-hot encoding and convert lists to numpy arrays
Y_train_three_digits = to_categorical(Y_train_three_digits)
X_train_three_digits = np.array(X_train_three_digits)
Y_train_three_digits = np.array(Y_train_three_digits)

# Split three-digit dataset into train/test (80%/20%)
x_train_three_digits, x_test_three_digits, y_train_three_digits, y_test_three_digits = train_test_split(
    X_train_three_digits, Y_train_three_digits, test_size=0.2, random_state=1
)

# Further split training data into train/validation (75%/25% of the 80%)
x_train_three_digits, x_val_three_digits, y_train_three_digits, y_val_three_digits = train_test_split(
    x_train_three_digits, y_train_three_digits, test_size=0.25, random_state=1
)

# Create TensorFlow datasets for three-digit subset
# Training dataset with shuffling
train_dataset_three_digits = tf.data.Dataset.from_tensor_slices((x_train_three_digits, y_train_three_digits))
train_dataset_three_digits = train_dataset_three_digits.shuffle(buffer_size=1024).batch(batch_size)

# Validation dataset for three-digit subset
val_dataset_three_digits = tf.data.Dataset.from_tensor_slices((x_val_three_digits, y_val_three_digits))
val_dataset_three_digits = val_dataset_three_digits.batch(batch_size)

# Test dataset for three-digit subset
test_dataset_three_digits = tf.data.Dataset.from_tensor_slices((x_test_three_digits, y_test_three_digits))
test_dataset_three_digits = test_dataset_three_digits.batch(batch_size)


#%% 
"""
how to select loss function

BinaryCrossentropy class: 
    
Use this cross-entropy loss for binary (0 or 1) classification applications. 
The loss function requires the following inputs:
y_true (true label): This is either 0 or 1.
y_pred (predicted value): 
This is the model's prediction, i.e, a single floating-point value which either represents a logit, 
(i.e, value in [-inf, inf] when from_logits=True) or a probability (i.e, value in [0., 1.] when 
from_logits=False). Recommended Usage: (set from_logits=True)

CategoricalCrossentropy class:
    
Computes the crossentropy loss between the labels and predictions. Use this crossentropy loss function when there are two
 or more label classes. We expect labels to be 
provided in a one_hot representation. If you want to provide labels as integers, please use 
SparseCategoricalCrossentropy loss. There should be # classes floating point values per feature.
    
SparseCategoricalCrossentropy class:

Computes the crossentropy loss between the labels and predictions.
Use this crossentropy loss function when there are two or more label classes. 
We expect labels to be provided as integers. If you want to provide labels using one-hot representation, 
please use CategoricalCrossentropy loss. There should be # classes floating point values per feature for y_pred 
and a single floating point value per feature for y_true.

binary_crossentropy function:
    
    
https://stackoverflow.com/questions/57253841/from-logits-true-and-from-logits-false-get-different-training-result-for-tf-loss
https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
"""
"""
#logits False
#Example  BCE class 

y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
print('BCE class, logit=False: ',bce(y_true, y_pred).numpy())

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
print('BCE class, logit=False, reduction=None: ',bce(y_true, y_pred).numpy())

#BCE function

loss = tf.keras.losses.binary_crossentropy(y_true, y_pred,from_logits=False)
assert loss.shape == (2,)
print('BCE function, logit=False ',loss.numpy())
print('BCE_function_logit=False',np.mean(loss.numpy()))

#logits True
#BCE Class

y_true = [[0, 10], [100, 0]]
y_pred = [[0.6, 7], [80, 0.6]]

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
print('BCE class, logit=True: ',bce(y_true, y_pred).numpy())

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
print('BCE class, logit=True, reduction=None: ',bce(y_true, y_pred).numpy())

#BCE function

loss = tf.keras.losses.binary_crossentropy(y_true, y_pred,from_logits=True)
assert loss.shape == (2,)
print('BCE function, logit=True ',loss.numpy())
print('BCE function, logit=True ',np.mean(loss.numpy()))

"""


#%% train simple model from scratch using CCE loss and SGD optimizer

#%% train simple model from scratch using CCE loss and SGD optimizer

# Design a simple feedforward neural network model
# Input layer with 784 neurons (flattened 28x28 MNIST images)
inputs = keras.Input(shape=(784,), name="digits")

# First hidden layer with 64 neurons and ReLU activation
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Second hidden layer with 64 neurons and ReLU activation 
x = layers.Dense(64, activation="relu", name="dense_2")(x)

# Output layer with 10 neurons (one for each digit 0-9)
# No activation specified since we're using from_logits=True in loss
outputs = layers.Dense(10, name="predictions")(x)

# Create the model by specifying inputs and outputs
model = keras.Model(inputs=inputs, outputs=outputs)

# Initialize SGD optimizer with learning rate of 0.001
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

# Use Categorical Cross Entropy loss since our labels are one-hot encoded
# from_logits=True since our final layer doesn't have softmax activation
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Initialize metrics to track training and validation accuracy
# Using CategoricalAccuracy since our labels are one-hot encoded
# (if using integer labels, would use SparseCategoricalAccuracy instead)
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

# Print model architecture summary
model.summary()

# Compile the model with our loss function, optimizer and metrics
model.compile(loss = loss_fn, 
            optimizer = optimizer,
            metrics=["accuracy"])

#%% train model on mnist from scratch for only one epoch for testing purpose

#%% Train model on MNIST for one epoch (for testing purposes)

# Import necessary callbacks
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Define path and model name for saving
path = "\\adversarial_reprog\\"
model_name = path + 'model_minist_epoch_1.h5'

# Create ModelCheckpoint callback
# Saves model when accuracy improves
checkp = ModelCheckpoint(model_name, 
                       monitor='accuracy',
                       save_best_only=True,
                       verbose=1)

# Create EarlyStopping callback
# Stops training if accuracy doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
   monitor="accuracy",
   patience=3
)

# Train the model for 1 epoch
# Using batch size of 64 and 10% of training data for validation
history = model.fit(x_train, 
                  y_train,
                  epochs=1,
                  batch_size=64,
                  validation_split=0.1,
                  callbacks=[checkp])

# Commented out alternative training configuration
#history = model_cora.fit(x_train_cora, y_train_cora, epochs = 100, batch_size = 32, validation_split=0.1)

# Plot training results
plt.figure(figsize=(20, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['MSE_train', 'MSE_validation'])
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Train vs Validation Loss')
plt.show()

# Commented out code for loading saved model
#from keras.models import load_model
#model_name='./model_cora_500_100_50.h5'
#model = load_model(model_name)

# Make predictions on validation set
pred_mnist = model.predict(x_val)
print(pred_mnist.shape)

# Evaluate model performance on validation set
score = model.evaluate(x_val, y_val, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#%%
"""
definitions, our base model is mnist with 10 output neurons, adversarial_model is mnist with only three outputs 
(0,1,2). 
"""

"""
Definitions:
- Base model (MNIST): 10 output neurons for digits 0-9
- Adversarial model: 3 output neurons for digits 0-1-2 only
"""

# Number of output classes for each model
base_model_output_length = 10  # Full MNIST (0-9)
adversarial_model_output_length = 3  # Subset (0-2)

# Define mapping between adversarial model classes and base model classes
# Example: 3 adversarial classes mapped to base model classes:
# Class 0 -> [0,1,2]
# Class 1 -> [3,4,5]
# Class 2 -> [6,7,8,9]
multi_label_mapping = [[0,1,2], [3,4,5], [6,7,8,9]]

# Flag to indicate if model outputs raw logits (True) or probabilities (False)
from_logits = True

# Initialize categorical cross entropy loss function
train_loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

def multi_label_loss_fn(y_true, y_pred, multi_label_mapping=multi_label_mapping, 
                      train_loss_function=train_loss_function):
   """
   Compute multi-label loss between true labels and predicted logits
   
   Args:
       y_true: True labels
       y_pred: Predicted logits from model
       multi_label_mapping: List of lists mapping adversarial classes to base model classes
       train_loss_function: Loss function to use (default: categorical cross entropy)
   
   Returns:
       Computed loss value
   """
   # Validate mapping length matches adversarial model output size
   if len(multi_label_mapping) != adversarial_model_output_length:
       raise Exception("multi_label_mapping length is not equal to base model output length")
   
   l = []
   # For each adversarial class
   for i in range(adversarial_model_output_length):
       # Get predictions for base model classes mapped to this adversarial class
       temp = y_pred.numpy()[:, multi_label_mapping[i]]
       # Take mean of predictions across mapped classes
       temp = temp.mean(axis=1)
       l.append(temp)
   
   # Convert to numpy array and transpose to match expected shape
   l1 = np.array(l)
   l1 = np.transpose(l1)
   
   # Return computed loss
   return train_loss_function(y_true, l1)

def multi_label_maping(y_pred, multi_label_mapping=multi_label_mapping):
   """
   Map predictions from base model space to adversarial model space
   
   Args:
       y_pred: Predicted logits from base model
       multi_label_mapping: List of lists mapping adversarial classes to base model classes
   
   Returns:
       Mapped predictions in adversarial model space
   """
   # Validate mapping length matches adversarial model output size
   if len(multi_label_mapping) != adversarial_model_output_length:
       raise Exception("multi_label_mapping length is not equal to base model output length")
   
   l = []
   # For each adversarial class
   for i in range(adversarial_model_output_length):
       # Get predictions for base model classes mapped to this adversarial class
       temp = y_pred.numpy()[:, multi_label_mapping[i]]
       # Take mean of predictions across mapped classes
       temp = temp.mean(axis=1)
       l.append(temp)
   
   # Convert to numpy array and transpose to match expected shape
   l1 = np.array(l)
   return np.transpose(l1)

#%% 
"""
load trained model only one epoch use it for testing purpose. We test MLM loss and zeroth order optimizer.

"""
from keras.models import load_model

#model_mnist = load_model('./model_minist_epoch_1.h5')

model_mnist = load_model(path+'model_minist_epoch_1.h5')

model_mnist.trainable = False # donot train its weights

#%%

import keras

inputs = keras.Input(shape=(784,), name="digits")
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.


x1 = layers.Dense(784, activation="sigmoid", name="reprogramed")(inputs)
#x2=keras.layers.Dense(1500, activation = 'sigmoid')(x1)
#x3=keras.layers.Dense(1433, activation = 'sigmoid')(x2)



outputs = model_mnist(x1, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
# A Dense classifier with a single unit (binary classification)
#outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

#%%
"""
save initial weights for comparison purpose
"""

# Store initial model weights for potential reset or comparison later
initial_weights = model.get_weights()

# Initialize SGD (Stochastic Gradient Descent) optimizer
# Learning rate set to 0.001 (1e-3)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

# Define the loss function
# Using CategoricalCrossentropy since labels are one-hot encoded
# from_logits=True since final layer outputs raw logits (no softmax)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Commented out alternative loss functions:
#loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # For integer labels
#loss_fn_binary = keras.losses.BinaryCrossentropy(from_logits=True)      # For binary classification

# Initialize accuracy metrics for training and validation
# Using CategoricalAccuracy since labels are one-hot encoded
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

# Display model architecture summary
model.summary()

# Compile the model with:
# - Defined loss function (CategoricalCrossentropy)
# - SGD optimizer
# - Accuracy metric for evaluation
model.compile(loss = loss_fn, 
            optimizer = optimizer,
            metrics=["accuracy"])


#%%

import tensorflow as tf
import numpy as np
import time

def train_zeroth_order_model(
    model,
    train_dataset,
    val_dataset,
    epochs=300,
    batch_size=32,
    q=2,
    beta=None,
    lay_no=1,
    learning_rate=1e-3,
    log_file="results.txt",
    saved_model_name='mnist_best_val_zeroth_MLM.h5'
):
    """
    Train a model using zeroth-order optimization method with multi-label mapping.
    
    Args:
        model: Keras model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        q: Number of times gradient is computed using random noise
        beta: Step size for gradient estimation (if None, calculated based on input size)
        lay_no: Layer number to treat as reprogramming layer
        learning_rate: Learning rate for optimizer
        log_file: File to log training progress
        saved_model_name: Name for saving best model
        
    Returns:
        tuple: (train_losses, val_losses, final_weights)
    """
    # Initialize metrics and optimizer
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Calculate beta if not provided
    if beta is None:
        input_size = train_dataset.element_spec[0].shape[1]
        beta = 1/input_size
    
    # Initialize tracking variables
    best_val_metric = 0
    val_loss_epoch_append = []
    train_loss_epoch_append = []
    
    # Log training parameters
    with open(log_file, "a") as f:
        f.write('\n')
        f.write(f"q={q}\n")
        f.write(f"beta={beta}\n")
        f.write(f"batch size={batch_size}\n")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()
        
        train_step_loss_append = []
        
        # Batch training loop
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Store current weights
            previous_weights = model.get_weights()
            weights = [layer.get_weights() for layer in model.layers]
            
            # Get initial prediction and loss
            with tf.GradientTape() as tape:
                logit = model(x_batch_train, training=True)
                loss = multi_label_loss_fn(y_batch_train, logit)
            
            # Compute zeroth-order gradient estimate
            g_append = []
            for i in range(q):
                # Generate random noise vector
                size = weights[lay_no][0].size + weights[lay_no][1].size
                n = np.random.randn(size)
                norm = np.linalg.norm(n)
                m = n/norm
                n = m * beta
                
                # Split noise vector into weights and bias
                w = weights[lay_no][0] + n[0:weights[lay_no][0].size].reshape(
                    weights[lay_no][0].shape[0], weights[lay_no][0].shape[1]
                )
                bias = weights[lay_no][1] + n[-weights[lay_no][1].size:]
                
                # Update layer weights temporarily
                l = [w.astype('float32'), bias.astype('float32')]
                model.get_layer('reprogramed').set_weights(l)
                
                # Get new prediction and loss
                logit_new = model(x_batch_train, training=True)
                loss_new = multi_label_loss_fn(y_batch_train, logit_new)
                
                # Calculate gradient estimate
                g = ((loss_new.numpy()-loss.numpy())/beta) * weights[lay_no][0].size
                g = g * m
                g_append.append(g)
            
            # Average gradient estimates
            g = np.mean(g_append, axis=0)
            
            # Restore original weights
            model.set_weights(previous_weights)
            
            # Prepare gradients for optimizer
            w_zero_order = g[0:weights[lay_no][0].size].reshape(
                weights[lay_no][0].shape[0], weights[lay_no][0].shape[1]
            ).astype('float32')
            bias_zero_order = g[-weights[lay_no][1].size:].astype('float32')
            
            grads_zero_order = [
                tf.convert_to_tensor(w_zero_order),
                tf.convert_to_tensor(bias_zero_order)
            ]
            
            # Apply gradients
            optimizer.apply_gradients(zip(grads_zero_order, model.trainable_weights))
            
            # Update metrics
            train_acc_metric.update_state(
                y_batch_train, 
                tf.convert_to_tensor(multi_label_maping(logit))
            )
            
            # Log progress
            with open(log_file, "a") as f:
                f.write(f'Training loss (for one batch) at step {step} loss:{float(loss)} epoch:{epoch}\n')
            
            if step % 2000 == 0:
                print(f"Training loss (for one batch) at step {step}: {float(loss):.4f}")
                train_step_loss_append.append(float(loss))
        
        # Calculate epoch metrics
        train_loss_epoch_append.append(sum(train_step_loss_append) / len(train_step_loss_append))
        train_acc = train_acc_metric.result()
        
        print(f"Training acc over epoch: {float(train_acc):.4f}")
        with open(log_file, "a") as f:
            f.write(f"Training acc over epoch:{train_acc} epoch:{epoch}\n")
        
        train_acc_metric.reset_states()
        
        # Validation loop
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = multi_label_loss_fn(y_batch_val, val_logits)
            val_acc_metric.update_state(y_batch_val, multi_label_maping(val_logits))
        
        val_acc = val_acc_metric.result()
        val_loss_epoch_append.append(float(val_loss_value))
        
        # Save model if validation accuracy improves
        if float(val_acc) > best_val_metric:
            best_val_metric = float(val_acc)
            model.save(saved_model_name, overwrite=True)
        
        val_acc_metric.reset_states()
        
        # Log validation metrics
        print(f"Validation acc: {float(val_acc):.4f}")
        print(f"Validation loss: {float(val_loss_value):.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"Validation acc: {float(val_acc)} epoch:{epoch}\n")
            f.write(f"Validation loss: {float(val_loss_value)} epoch:{epoch}\n\n")
    
    return train_loss_epoch_append, val_loss_epoch_append, model
    
    
    #print("Time taken: %.2fs" % (time.time() - start_time))
final_weights=model.get_weights()


#%%

# First, let's make sure we have all dependencies loaded
import tensorflow as tf
import numpy as np
import time

# Initialize parameters for training
EPOCHS = 300
BATCH_SIZE = 4
Q = 2  # number of gradient estimations
LAY_NO = 1  # reprogramming layer number
LEARNING_RATE = 1e-3
SAVE_PATH = 'mnist_best_val_zeroth_MLM.h5'
LOG_FILE = 'results_zeroth_order.txt'

# Call the training function
print("Starting zeroth-order training...")
train_losses, val_losses, final_model = train_zeroth_order_model(
    model=model,
    train_dataset=train_dataset_three_digits,
    val_dataset=val_dataset_three_digits,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    q=Q,
    lay_no=LAY_NO,
    learning_rate=LEARNING_RATE,
    log_file=LOG_FILE,
    saved_model_name=SAVE_PATH
)

# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Print final metrics
print("\nTraining completed!")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print(f"Best model saved to: {SAVE_PATH}")


#%% testing the model

score = final_model.evaluate(x_val, y_val, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


    
#%%
  