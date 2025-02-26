from classifiers.random_forest_classifier import RandomForestMnistClassifier
from classifiers.nn_classifier import NNMnistClassifier
from classifiers.cnn_classifier import CNNMnistClassifier

class MnistClassifier:
    """
    A class that selects and uses the appropriate MNIST classifier based on the algorithm parameter.
    Algorithm values:
        - 'rf': Random Forest
        - 'nn': Feed-Forward Neural Network
        - 'cnn': Convolutional Neural Network
    """

    def __init__(self, algorithm, **kwargs):
        algorithm = algorithm.lower()
        if algorithm == 'rf':
            self.classifier = RandomForestMnistClassifier(**kwargs)
        elif algorithm == 'nn':
            self.classifier = NNMnistClassifier(**kwargs)
        elif algorithm == 'cnn':
            self.classifier = CNNMnistClassifier(**kwargs)
        else:
            raise ValueError("Algorithm not recognized. Use 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        """
        Train the selected classifier.
        """
        self.classifier.train(X_train, y_train)

    def predict(self, X):
        """
        Predict labels using the selected classifier.
        """
        return self.classifier.predict(X)
