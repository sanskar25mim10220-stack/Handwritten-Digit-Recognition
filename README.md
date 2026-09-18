# Handwritten-Digit-Recognition

README — Handwritten Digit Recognition using CNN (MNIST Dataset) Project Overview

Handwritten digit recognition is a classic machine learning problem that involves classifying digits (0–9) from images. This project uses the MNIST dataset and a Convolutional Neural Network (CNN) model to automatically identify handwritten digits with high accuracy.

This project is ideal for beginners learning the fundamentals of AI / ML / Deep Learning.

Features

Recognizes digits (0–9) from handwritten images

Uses a simple and efficient CNN architecture

Achieves 98–99% accuracy

Beginner-friendly implementation using TensorFlow/Keras

Includes visualization and prediction on sample test images

Dataset MNIST Dataset

70,000 grayscale images (28×28 pixels)

60,000 training images

10,000 testing images

It is automatically loaded from Keras, so no manual download is needed.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

Model Architecture

The CNN model includes:

Conv2D + MaxPooling layers

Flatten layer

Dense layer (ReLU)

Output layer (Softmax)

This architecture extracts spatial features and classifies the digit effectively.

Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Jupyter Notebook / Google Colab

The model will:
Train on MNIST Display accuracy and loss Predict a digit from the test set

Results

Achieved accuracy: ~99% on test data

Model correctly predicts handwritten digits

Training and validation losses decrease over epochs

📸 Sample Prediction Code plt.imshow(x_test[0].reshape(28,28), cmap='gray') plt.show() prediction = model.predict(x_test[0].reshape(1,28,28,1)) print("Predicted Digit:", prediction.argmax())

Future Improvements

Add data augmentation

Deploy as a web app using Streamlit/Flask

Train on EMNIST for alphabets + digits

Add GUI for drawing digits

Applications

Postal code reading

Bank cheque digit recognition

Form/Document digitization

Automated data entry

Contributions

Pull requests, suggestions, or improvements are welcome!
