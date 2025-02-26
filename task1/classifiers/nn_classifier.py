from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from .mnist_classifier_interface import MnistClassifierInterface

class NNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using a simple feed-forward neural network.
    """

    def __init__(self, input_shape=(28, 28), num_classes=10, epochs=5, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        # Build a simple feed-forward neural network.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train):
        # Normalize input data
        X_train = X_train / 255.0
        # Convert labels to one-hot encoding.
        y_train_cat = to_categorical(y_train, self.num_classes)
        history = self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        print("Feed-Forward NN training complete.")

        # Plot loss and accuracy
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
        # Normalize input data
        X = X / 255.0
        # Predict
        preds = self.model.predict(X)
        # Return the index of the highest probability.
        return preds.argmax(axis=1)