
# TensorFlow Demo Notice

## Issue with Python 3.13
The `ml-demo.py` file requires TensorFlow, but TensorFlow doesn't currently support Python 3.13.

## Solutions

### Option 1: Skip the TensorFlow Demo
The TensorFlow demo (`ml-demo.py`) is optional and not required for the main functionality of this project. You can skip running it.

### Option 2: Use Python 3.11 or Earlier
If you need to run the TensorFlow demo:

1. Install Python 3.11 using pyenv:
   ```bash
   pyenv install 3.11.10
   pyenv local 3.11.10
   ```

2. Create a new virtual environment:
   ```bash
   python -m venv venv_tf
   source venv_tf/bin/activate  # On Windows: venv_tf\Scripts\activate
   ```

3. Install requirements with TensorFlow:
   ```bash
   # Uncomment the TensorFlow lines in requirements.txt first
   pip install -r requirements.txt
   ```

4. Run the demo:
   ```bash
   python demos/ml-demo.py
   ```

### Option 3: Use Google Colab
Upload `ml-demo.py` to Google Colab which has TensorFlow pre-installed and supports the latest versions.

## Demo Description
The `ml-demo.py` demonstrates:
- Loading the MNIST handwritten digits dataset
- Creating a neural network with TensorFlow/Keras
- Training the model to recognize digits
- Evaluating model performance
- Making predictions on test images
