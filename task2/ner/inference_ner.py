import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def extract_animal_entities(text, model_dir="./ner/ner_model"):
    """
    Loads the trained NER model and extracts animal entities from the input text.
    """
    # Load trained NER model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Create a NER pipeline using the loaded model and tokenizer.
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    # Return the extracted entities.
    results = ner_pipeline(text)
    return results

if __name__ == "__main__":
    # Create an ArgumentParser to handle command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./ner/ner_model")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    extract_animal_entities(args.text, args.model_dir)