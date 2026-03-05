# models/predictor.py

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ---------- 工单分类预测 ----------
clf_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
clf_model = BertForSequenceClassification.from_pretrained("models/bert_classifier")
clf_model.eval()

# 自动读取训练时保存的标签映射
clf_id2label = clf_model.config.id2label


def predict_classifier(text: str):
    inputs = clf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return clf_id2label[predicted_class_id]


# ---------- 情感分析预测 ----------
sent_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
sent_model = BertForSequenceClassification.from_pretrained("models/bert_sentiment")
sent_model.eval()

sent_id2label = sent_model.config.id2label


def predict_sentiment(text: str):
    inputs = sent_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = sent_model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return sent_id2label[predicted_class_id]
