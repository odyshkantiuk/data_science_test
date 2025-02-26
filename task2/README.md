# Task 2. Named entity recognition + image classification

Task:

1. The user provides a text similar to “There is a cow in the picture.” and an image that
contains any animal.
2. Pipeline decide if it is true or not and provide a boolean value as the output.


## Project structure

The repository is organized as follows:

- [image_classifier](image_classifier)
  - [image_dataset](image_classifier/image_dataset)
  - [animal_classifier_model.h5](image_classifier/animal_classifier_model.h5)
  - [inference_image_classifier.py](image_classifier/inference_image_classifier.py)
  - [train_image_classifier.py](image_classifier/train_image_classifier.py)
- [ner](ner)
  - [ner_model](ner/ner_model)
  - [animal_ner_dataset_train.json](ner/animal_ner_dataset_train.json)
  - [animal_ner_dataset_val.json](ner/animal_ner_dataset_val.json)
  - [inference_ner.py](ner/inference_ner.py)
  - [train_ner.py](ner/train_ner.py)
- [demo.ipynb](demo.ipynb)
- [eda.ipynb](eda.ipynb)
- [pipeline.py](pipeline.py)
- [README.md](README.md)
- [requirements.txt](requirements.txt)


## Setup instructions

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/odyshkantiuk/data_science_test.git
    cd data_science_test/task2
    ```

2. **Create conda env:**

   [Installed tensorflow version:](https://www.tensorflow.org/install/pip): the latest release of TensorFlow that supported GPUs on native Windows

   [Installed torch version:](https://pytorch.org/get-started/locally/) `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` 

   Create a new conda environment:
    ```bash
    conda create --name <env> --file requirements.txt
    ```
   Activate the conda environment:
    ```bash
    conda activate <env>
    ```
3. **Using the task solution:**
    
   1. Create datasets: run [eda.ipynb](eda.ipynb)
   2. Run demonstration: run [demo.ipynb](demo.ipynb)
