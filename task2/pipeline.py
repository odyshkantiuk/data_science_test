import argparse
from ner.inference_ner import extract_animal_entities
from image_classifier.inference_image_classifier import predict_animal_label

def pipeline(text, image_path, ner_model_dir="./ner/ner_model", classifier_model_path="./image_classifier/animal_classifier_model.h5", img_height=227, img_width=227):
    """
    Runs the full pipeline:
      1. Extract animal entities from the text using the NER model.
      2. Classify the image to predict the animal label.
      3. Returns True if the predicted label is an extracted animal.
    """
    print("Running NER on text...")
    animal_entities = extract_animal_entities(text, model_dir=ner_model_dir)
    print("Animals mentioned in text:", animal_entities)

    print("Running image classification on image...")
    predicted_label = predict_animal_label(image_path, classifier_model_path, img_height, img_width)
    print("Animal predicted from image:", predicted_label)

    result = predicted_label in animal_entities[0]['word'] if predicted_label is not None else False
    print("Pipeline result (True if match, False otherwise):", result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--ner_model_dir", type=str, default="./ner/ner_model")
    parser.add_argument("--classifier_model_path", type=str, default="./image_classifier/animal_classifier_model.h5")
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    args = parser.parse_args()
    pipeline(args.text, args.image_path, args.ner_model_dir, args.classifier_model_path, args.img_height, args.img_width)
