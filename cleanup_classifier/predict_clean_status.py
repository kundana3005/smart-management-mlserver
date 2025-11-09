import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model("model/cleanup_accuracy_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

classes = ["Cleaned", "Uncleaned"]

test_folder = "test_images"

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0][0] > 0.5)
        print(f"{img_name} → {classes[predicted_class]}")