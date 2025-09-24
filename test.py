import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('models/imageClassifier1.h5')

# Load and preprocess the image
img_path = "dog.webp"   # change this to your test image
img = cv2.imread(img_path)

# Convert BGR (OpenCV default) to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to model input size
resize = tf.image.resize(img_rgb, (256,256))

# Make prediction
yhat = model.predict(np.expand_dims(resize/255, 0))

# Interpret result (binary classifier: cat vs dog)
if yhat > 0.5: 
    print("Predicted class: Dog")
else:
    print("Predicted class: Cat")
