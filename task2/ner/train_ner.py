import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, IntervalStrategy
from datasets import Dataset
import json

def train_ner(output_dir="./ner/ner_model", epochs=5, batch_size=16, learning_rate=5e-5):
    """
    Training a NER model based on a created dataset.
    """
    # Load dataset for training.
    def read_dataset(path):
        with open(path, "r") as f:
            loaded_dataset_dict = json.load(f)
        return Dataset.from_dict(loaded_dataset_dict)
    dataset_train = read_dataset("./ner/animal_ner_dataset_train.json")
    dataset_val = read_dataset("./ner/animal_ner_dataset_val.json")

    # Define labels.
    label_list = ["O", "B-ANIMAL"]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    # Load model and tokenizer.
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize dataset.
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                label_ids.append(tags[word_idx] if word_idx is not None else -100)
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized
    tokenized_dataset_train = dataset_train.map(tokenize_and_align, batched=True)
    tokenized_dataset_val = dataset_val.map(tokenize_and_align, batched=True)

    # Setup training arguments and trainer.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy=IntervalStrategy.EPOCH,
        logging_dir="./ner/logs",
        no_cuda=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val
    )

    # Train and save.
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Create an ArgumentParser to handle command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./ner/ner_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    train_ner(args.output_dir, args.epochs, args.batch_size, args.learning_rate)
