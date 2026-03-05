# main.py

from models.classifier import train_classifier_model
from models.predictor import predict_classifier, predict_sentiment
from models.sentiment import train_sentiment_model
from models.reply_llm_local import stream_generate_reply


def run_classifier_demo():
    print("开始训练工单分类模型...")
    # train_classifier_model()
    test_text = "期待您的坏消息！"
    label = predict_classifier(test_text)
    print(f"【{test_text}】 => 分类结果: {label}")


def run_sentiment_demo():
    print("开始训练情感分析模型...")
    # train_sentiment_model()
    test_text = "这质量也太差了吧，退货都没人理"
    result = predict_sentiment(test_text)
    print(f"【{test_text}】 => 情感结果: {result}")


def run_llm_demo():
    print("智能回复生成中(By:Qwen1.5-1.8B-Chat)...")
    text = "耳机买了一个星期都没发货，客服也联系不上"
    category = predict_classifier(text)
    sentiment = predict_sentiment(text)
    stream_generate_reply(text, category, sentiment)


if __name__ == "__main__":
    # 可选择运行哪一个模块
    run_classifier_demo()
    # run_sentiment_demo()
    # run_llm_demo()
