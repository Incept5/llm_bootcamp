
# Machine Learning Demo

This directory contains machine learning examples and demonstrations using TensorFlow and other ML frameworks.

## Files

### ml-demo.py

A comprehensive neural network demonstration using TensorFlow and the MNIST handwritten digit dataset.

#### What it does:

This script implements a complete machine learning pipeline for handwritten digit recognition:

1. **Data Loading & Preprocessing**
   - Loads the famous MNIST dataset (70,000 images of handwritten digits 0-9)
   - Normalizes pixel values from 0-255 range to 0-1 range for better training performance
   - Splits data into training (60,000 images) and test (10,000 images) sets

2. **Neural Network Architecture**
   - **Input Layer**: Accepts 28x28 pixel images
   - **Flatten Layer**: Converts 2D image data into 1D array (784 features)
   - **Hidden Layer**: Dense layer with 128 neurons and ReLU activation function
   - **Dropout Layer**: 20% dropout rate to prevent overfitting
   - **Output Layer**: 10 neurons (one for each digit 0-9) with linear activation

3. **Model Training**
   - Uses Adam optimizer for efficient gradient descent
   - Employs Sparse Categorical Crossentropy loss function
   - Tracks accuracy metric during training
   - Trains for 5 epochs

4. **Model Evaluation**
   - Tests the trained model on unseen test data
   - Reports final test accuracy
   - Typically achieves 97-98% accuracy

5. **Prediction Demonstration**
   - Creates a probability model with softmax activation for predictions
   - Makes a prediction on a test image
   - Displays the test image and predicted digit using matplotlib

#### Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib

#### Usage

```bash
python ml-demo.py
```

The script will:
1. Download the MNIST dataset automatically (if not already cached)
2. Train the neural network (takes 1-2 minutes)
3. Display test accuracy results
4. Show a sample image with its predicted label

#### Key Concepts Demonstrated

- **Deep Learning**: Multi-layer neural network
- **Image Classification**: Converting images to numerical predictions
- **Data Preprocessing**: Normalization and reshaping
- **Model Evaluation**: Separating training and testing data
- **Overfitting Prevention**: Using dropout layers
- **Visualization**: Displaying results with matplotlib

This is an excellent starting point for understanding fundamental machine learning concepts and TensorFlow usage.
