import argparse

import datasets
from tf3.training_encoder.labels import get_labels
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint", type=str, default="bert-base-multilingual-cased"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="training_encoder/ner_dataset_tokenized/"
    )
    parser.add_argument("--output_dir", type=str, default="bert-ner-model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    tokenized_datasets = datasets.load_from_disk(args.dataset_path)
    train_dataset, eval_dataset = tokenized_datasets.train_test_split(
        test_size=0.1
    ).values()

    labels, label2id, id2label = get_labels()

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        save_total_limit=2,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="wandb",
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
