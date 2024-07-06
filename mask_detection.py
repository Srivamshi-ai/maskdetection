import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Initialize the CNN model
cnn = Sequential([
    Conv2D(filters=100, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=100, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(35, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile the model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Labels and color dictionary for mask detection
labels_dict = {0: 'No mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Set image resize factor
imgsize = 4

# Initialize camera
camera = cv2.VideoCapture(0)

# Load the pre-trained face detection model
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    (rval, im) = camera.read()
    im = cv2.flip(im, 1, 1)  # Mirror the image

    # Resize the image for faster processing
    imgs = cv2.resize(im, (im.shape[1] // imgsize, im.shape[0] // imgsize))

    # Detect faces in the image
    face_rec = classifier.detectMultiScale(imgs)

    for i in face_rec:
        (x, y, l, w) = [v * imgsize for v in i]  # Scale back the face coordinates
        face_img = im[y:y + w, x:x + l]

        # Preprocess the face image
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])

        # Predict mask or no mask
        result = cnn.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw rectangle and label around the face
        cv2.rectangle(im, (x, y), (x + l, y + w), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + l, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    # Break the loop on 'Esc' key press
    if key == 27:
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
