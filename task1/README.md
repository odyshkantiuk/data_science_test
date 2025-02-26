# Task 1. Image classification + OOP

This task demonstrates the implementation of three MNIST classifiers using OOP. The solution uses three different models:

- **Random Forest (RF)**
- **Feed-Forward Neural Network (NN)**
- **Convolutional Neural Network (CNN)**

Each classifier implements a common interface [MnistClassifierInterface](classifiers/mnist_classifier_interface.py) with two abstract methods: *train* and *predict*. [MnistClassifier](mnist_classifier.py) takes as an input parameter the name of the algorithm and provides predictions with exactly the same structure (inputs and outputs) not depending on the selected algorithm.

## Project structure

The repository is organized as follows:

- [classifiers](classifiers)
  - [\_\_init\_\_.py](classifiers/__init__.py)
  - [cnn_classifier.py](classifiers/cnn_classifier.py)
  - [mnist_classifier_interface.py](classifiers/mnist_classifier_interface.py)
  - [nn_classifier.py](classifiers/nn_classifier.py)
  - [random_forest_classifier.py](classifiers/random_forest_classifier.py)
- [demo.ipynb](demo.ipynb)
- [mnist_classifier.py](mnist_classifier.py)
- [README.md](README.md)
- [requirements.txt](requirements.txt)


## Setup instructions

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/odyshkantiuk/data_science_test.git
    cd data_science_test/task1
    ```

2. **Create a virtual environment:**
    
    For Linux/macOS:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    For Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. **Using conda:**

   Create a new conda environment:
    ```bash
    conda create --name env_name python=3.8.20
    ```
   Activate the conda environment:
    ```bash
    conda activate env_name
    ```

4. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
