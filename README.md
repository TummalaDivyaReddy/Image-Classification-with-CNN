# Image-Classification-with-CNN

code file: [Image-Classification-with-CNN](Image_Classification_with_CNN.ipynb)

Convolutional Neural Networks (CNNs) are a type of deep learning algorithm that are commonly used for image classification.

The CIFAR-10 dataset is a collection of 60,000 small color images of 10 different classes. The goal of the CNN is to learn to classify these images into their respective classes. 

The CNN architecture consists of a series of convolution layers, pooling layers, and fully connected layers. Convolution layers are used to extract features from the images, pooling layers are used to reduce the size of the feature maps, and fully connected layers are used to classify the images. 

The CNN is trained using a technique called stochastic gradient descent (SGD). SGD is an iterative optimization algorithm that updates the weights of the CNN's layers based on the error between the predicted and actual labels. 

The CNN is evaluated on a held-out test set of images. The accuracy of the CNN is reported as the percentage of images that are correctly classified.

## Follow the instructions and then report how the performance changed.
It is an implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

```ruby
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Create the model
model = Sequential()


# Convolutional input layer, 32 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=max_norm(3)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Convolutional layer, 32 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
# Max Pool layer with size 2×2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
# Max Pool layer with size 2×2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Convolutional layer, 128 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Convolutional layer, 128 feature maps with a size of 3×3 and a rectifier activation function
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
# Max Pool layer with size 2×2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten layer
model.add(Flatten())
# Dropout layer at 20%
model.add(Dropout(0.2))


# Fully connected layer with 1024 units and a rectifier activation function
model.add(Dense(1024, activation='relu', kernel_constraint=max_norm(3)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Fully connected layer with 512 units and a rectifier activation function
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
# Dropout layer at 20%
model.add(Dropout(0.2))


# Fully connected output layer with 10 units and a Softmax activation function
model.add(Dense(num_classes, activation='softmax'))


# Compile model
epochs = 25
learning_rate = 0.01
decay = learning_rate / epochs
sgd = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

```
The first line imports the NumPy library, which is used for scientific computing. The next two lines import the CIFAR-10 dataset and the Keras library. Keras is a high-level neural network API that makes it easy to build and train neural networks.matplotlib.pyplot is used for data visualization.
The seed and np.random.seed(seed) ensure reproducibility by fixing the random number generator's seed.
The next few lines load the data and normalize it. This means that the values of each pixel in the images are scaled from 0 to 1. This is necessary for the neural network to work properly.
The model is created using the Sequential API from Keras, which allows us to stack layers one after another and it consists of a series of convolutional layers, max-pooling layers, and fully connected (dense) layers.
The Conv2D layers apply convolutional filters to extract features from the input images, maxPooling2D layers downsample the feature maps to reduce spatial dimensions, flatten layer is used to convert the 2D feature maps into a 1D vector before feeding it to the dense layers, dense layers are fully connected layers that perform classification based on the features learned by the previous layers,dropout layers are used for regularization to prevent overfitting.
For compilation the SGD optimizer (Stochastic Gradient Descent) is used for categorical cross-entropy loss function and accuracy as the evaluation metric.
The model is trained using the fit method on the training data and the history object stores the training and validation loss and accuracy for each epoch.
The model is evaluated on the test data to get the final accuracy.

## Did the performance change?

YES! The performance has changed. From the output I observe that there is a Decrease in loss and increase in the accuracy respective to each epoch and even has an overall increase in accuracy.

## Predict the first 4 images of the test data using the above model. Then, compare with the actual label for those 4 images to check whether or not the model has predicted correctly.

```ruby
# Predict the first 4 images in the test data
predictions = model.predict(X_test[:4])
predicted_labels = np.argmax(predictions, axis=1)


# Actual labels for the first 4 images
actual_labels = np.argmax(y_test[:4], axis=1)


print("Predicted Labels:", predicted_labels)
print("Actual Labels:", actual_labels)

```

The model is used to predict the class labels for the first 4 test images and the predicted labels are then compared with the actual labels to check the model's accuracy on these samples.

## Visualize Loss and Accuracy using the history object

```ruby
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

```

The training and validation loss over each epoch are plotted using matplotlib.pyplot. Similarly, the training and validation accuracy over each epoch are also plotted for visualization.


[Youtube link:] ()
