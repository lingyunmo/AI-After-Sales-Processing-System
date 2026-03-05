# sentiment.py
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# 加载数据并编码 label
def load_data():
    df = pd.read_csv("data/sentiment_dataset.csv")
    label2id = {'negative': 0, 'neutral': 1, 'positive': 2}  # 固定顺序
    df["label"] = df["label"].map(label2id)
    return Dataset.from_pandas(df), label2id

# 训练函数
def train_sentiment_model():
    dataset, label2id = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

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
        output_dir="./results_sentiment",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10,
        logging_dir="./logs_sentiment"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    trainer.train()
    model.save_pretrained("models/bert_sentiment")
    print("情感分析模型训练完成并已保存！")
