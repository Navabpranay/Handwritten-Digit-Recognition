import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read")

    img = cv2.resize(img, (28, 28))
    # img = 255 - img  # Uncomment if your image digits are black on white background

    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_digit(image_path, model_path='mnist_cnn.h5'):

    model = load_model(model_path)  # Load the trained model from file

    processed_img = preprocess_image(image_path)

    prediction = model.predict(processed_img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

# Example usage:
image_path = 'images/test2.png'  # Replace with your image path
predicted_digit, confidence = predict_digit(image_path)
print(f"Predicted Digit: {predicted_digit} with confidence {confidence:.4f}")


'''
# to display the image the is seen by the system or the model
import matplotlib.pyplot as plt
processed_img = preprocess_image(image_path)  # Shape: (1, 28, 28, 1)
plt.imshow(processed_img[0, :, :, 0], cmap='gray')
plt.show()'''

