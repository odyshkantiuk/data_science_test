import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from imutils import paths
import random
import os
import cv2
import numpy as np

def train_image_classifier(dataset_dir="./image_classifier/image_dataset/raw-img", output_model="./image_classifier/animal_classifier_model.h5", img_height=227, img_width=227, batch_size=32, epochs=50):
    """
    Trains a simple CNN on an animal image dataset.
    """

    # Initialize lists for image data and labels.
    data = []
    labels = []

    # Get all image paths and shuffle them for randomness.
    image_paths = sorted(list(paths.list_images(dataset_dir)))
    random.seed(1)
    random.shuffle(image_paths)
    animals = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "ragno", "sheep", "squirrel"]
    animal_images = {}

    # Create a list of images and labels for animal.
    for animal in animals:
        animal_images[animal] = []
    for imagePath in image_paths:
        label = imagePath.split(os.path.sep)[-2]
        if label in animals:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (img_height, img_width))
            animal_images[label].append(image)
            data.append(image)
            labels.append(label)

    # Data normalization.
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # We divide the data into training and test samples.
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    # Performing One-Hot Encoding.
    encoder = OneHotEncoder(categories=[animals])
    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

    # Data augmentation.
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    # Creating a convolutional neural network AlexNet.
    model = Sequential([
        Input(shape=(img_height, img_width, 3)),
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Second block
        Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Third block
        Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),

        # Fourth block
        Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),

        # Fifth block
        Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Flatten
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(len(animals), activation='softmax')
    ])

    # Compile the neural network.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Set up a checkpoint to save the best model.
    checkpoint = ModelCheckpoint(output_model, save_best_only=True, monitor="val_accuracy", mode="max")

    # Train the model.
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    print("Image classifier training complete: ", output_model)

    # Loss schedule.
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.legend()

    # Accuracy graph.
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Validation")
    plt.legend()

    # Graph display.
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create an ArgumentParser to handle command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./image_classifier/image_dataset/raw-img")
    parser.add_argument("--output_model", type=str, default="./image_classifier/animal_classifier_model.h5")
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_image_classifier(args.dataset_dir, args.output_model, args.img_height, args.img_width, args.batch_size, args.epochs)
