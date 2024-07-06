# Real-Time Face Mask Detection

This project uses a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask in real-time using a webcam. The model is built using TensorFlow and Keras, and it uses OpenCV for video capture and face detection.

## Features

- Real-time face detection using Haar Cascade Classifier.
- Face mask detection using a CNN model.
- Visual indicators for mask and no mask detection.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/face-mask-detection.git
    cd face-mask-detection
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Haar Cascade XML file**:
    The `haarcascade_frontalface_default.xml` file should be included in the `cv2.data.haarcascades` path by default. If not, you can download it from [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Usage

1. **Run the script**:
    ```bash
    python mask_detection.py
    ```

2. **Interact with the application**:
    - The webcam will start and detect faces in real-time.
    - The application will draw a rectangle around detected faces and label them as "Mask" or "No mask" with different colors.

## Files

- `mask_detection.py`: The main script for running the face mask detection.
- `requirements.txt`: List of required Python packages.

## Model Training (Optional)

If you want to train the model yourself, you will need a dataset of images with and without masks. You can then modify the script to include model training steps.

## Acknowledgements

- The project uses the Haar Cascade Classifier for face detection from OpenCV.
- The CNN model is built using TensorFlow and Keras.
