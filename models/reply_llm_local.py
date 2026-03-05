# models/reply_llm_local.py（终止机制 + 打字机风格输出）

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True).to(device)
model.eval()

# 可自定义的“结束关键词”，模型生成这些词时提前终止
STOP_WORDS = ["祝您生活愉快。", "感谢您的理解。", "感谢您的支持。", "如有疑问,欢迎联系。"]

def stream_generate_reply(user_input: str, category: str, sentiment: str, max_new_tokens=128) -> str:
    """
    一次性生成完整客服回复，适配网页前端调用。
    """
    prompt = f"""你是一名电商售后客服，请根据用户的问题内容、工单分类与情感状态，生成一段语气温和、正式简洁的中文回复（不超过三句话）。使用纯文本给出。结束的时候使用以下关键词结束回复：“祝您生活愉快。", "感谢您的理解。", "感谢您的支持。", "如有疑问,欢迎联系。"”。

用户问题：{user_input}
工单分类：{category}
情感状态：{sentiment}
客服回复："""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取“客服回复”之后的内容
    reply = generated_text.split("客服回复：")[-1].strip()

    # 可选：根据结束语提前截断
    for stop_word in STOP_WORDS:
        if stop_word in reply:
            reply = reply.split(stop_word)[0] + stop_word
            break

    return reply
