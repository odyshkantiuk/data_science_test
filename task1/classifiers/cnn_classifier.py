from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from .mnist_classifier_interface import MnistClassifierInterface

class CNNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using a Convolutional Neural Network.
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10, epochs=5, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        # Build a CNN.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train):
        # Normalize input data.
        X_train = X_train / 255.0
        # Ensure data has the correct dimensions (num_samples, 28, 28, 1).
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(-1, self.input_shape[0], self.input_shape[1], 1)
        # Convert labels to one-hot encoding.
        y_train_cat = to_categorical(y_train, self.num_classes)
        # Train model
        history = self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        print("CNN training complete.")

        # Plot loss and accuracy.
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def predict(self, X):
        # Normalize input data.
        X = X / 255.0
        # Ensure data has the correct dimensions (num_samples, 28, 28, 1).
        if len(X.shape) == 3:
            X = X.reshape(-1, self.input_shape[0], self.input_shape[1], 1)
        # Predict.
        preds = self.model.predict(X)
        return preds.argmax(axis=1)
