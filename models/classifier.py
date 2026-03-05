import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


def load_data():
    df = pd.read_csv("data/network_shopping_tickets.csv")

    # 标签映射：logistics -> 0, complaint -> 1, ...
    label2id = {label: idx for idx, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label2id)

    return Dataset.from_pandas(df), label2id

def train_classifier_model():
    dataset, label2id = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_fn)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=len(label2id),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    trainer.train()
    model.save_pretrained("models/bert_classifier")
    print("工单分类模型训练完成并已保存！")