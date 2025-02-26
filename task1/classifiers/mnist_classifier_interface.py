from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Abstract interface for MNIST classifiers.
    Each classifier must implement the train and predict methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the classifier on the training data.
        :param X_train: Training images.
        :param y_train: Training labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict labels for the given images.
        :param X: Images to classify.
        :return: Predicted labels.
        """
        pass
