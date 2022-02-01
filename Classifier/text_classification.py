import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import os


# function for computing accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if type(predictions) == tuple:
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    acc = np.mean(predictions == labels)
    return {
        'accuracy': acc
    }

def load_data(args):   
    if args.dataset == "ag_news":
        dataset = load_dataset("ag_news")
        num_labels = 4
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_polarity")
        num_labels = 2
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        num_labels = 3

    dataset = dataset.shuffle(seed=0)

    return dataset, num_labels

def main(args):
    
    dataset, num_labels = load_data(args)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
    
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if args.dataset == "mnli":
        # only evaluate on matched validation set
        testset_key = "validation_matched"
        preprocess_function = lambda examples: tokenizer(
            examples["premise"], examples["hypothesis"], max_length=256, truncation=True)
    else:
        text_key = 'text' 
        testset_key = 'test'
        preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    train_args = TrainingArguments(
        args.checkpoint_folder,
        evaluation_strategy = "steps",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[testset_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    

    trainer.train()
    trainer.evaluate()
    suffix = '_finetune'
    torch.save(model.state_dict(),
               os.path.join(args.result_folder, "%s_%s%s.pth" % ('gpt2', args.dataset, suffix)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training gpt2 for text classification.")

    # Bookkeeping
    parser.add_argument("--checkpoint_folder", default="checkpoint/", type=str,
        help="folder in which to store temporary model checkpoints")
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder in which to store trained models")

    # Data 
    parser.add_argument("--dataset", default="ag_news", type=str,
        choices=["ag_news", "yelp", "mnli"],
        help="classification dataset to use")

    # Optimization
    parser.add_argument("--batch_size", default=16, type=int,
        help="batch size for training and evaluation")
    parser.add_argument("--epochs", default=5, type=int,
        help="number of epochs to train for")
    parser.add_argument("--lr", default=2e-5, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="weight decay")

    args = parser.parse_args()

    if args.result_folder == 'none':
        args.result_folder = args.checkpoint_folder

    main(args)
