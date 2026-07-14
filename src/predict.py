import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224   
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   


def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def predict(image_path, model_path="artifacts/best_model.h5"):

    model = load_trained_model(model_path)

    processed_img = preprocess_image(image_path)

    preds = model.predict(processed_img)

    class_index = np.argmax(preds)
    confidence = preds[0][class_index]

    print("Prediction:", CLASS_NAMES[class_index])
    print("Confidence:", confidence)
