from sklearn.ensemble import RandomForestClassifier
from .mnist_classifier_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using Random Forest.
    """

    def __init__(self, n_estimators=100):
        # Initialize the Random Forest model.
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)

    def train(self, X_train, y_train):
        # Random Forest in scikit-learn requires 2D input.
        # Flatten the images: from (num_samples, height, width) to (num_samples, height*width)
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        self.model.fit(X_train_flat, y_train)
        print("Random Forest training complete.")

    def predict(self, X):
        # Flatten the input images.
        X_flat = X.reshape((X.shape[0], -1))
        predictions = self.model.predict(X_flat)
        return predictions
