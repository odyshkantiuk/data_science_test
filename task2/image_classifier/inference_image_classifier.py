import argparse
import numpy as np
from tensorflow.keras.models import load_model
import cv2

def predict_animal_label(image_path="image.jpg", model_path="./image_classifier/animal_classifier_model.h5", img_height=227, img_width=227):
    """
    Loads the trained image classifier model and predicts the animal label from the image.
    """
    animals = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "ragno", "sheep", "squirrel"]

    # Load the pre-trained model.
    model = load_model(model_path)

    # Load and preprocess the image.
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_height, img_width))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction.
    predictions = model.predict(image)
    predicted_animal = animals[np.argmax(predictions)]
    return predicted_animal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="image.jpg")
    parser.add_argument("--model_path", type=str, default="./image_classifier/animal_classifier_model_1.h5")
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    args = parser.parse_args()
    predict_animal_label(args.image_path, args.model_path, args.img_height, args.img_width)
