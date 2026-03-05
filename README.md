# Intelligent After-Sales Customer Service System

This repository contains the core source code for my undergraduate Final Year Project (Thesis) in Computer Science.

## Project Overview

This project is an intelligent after-sales customer service system designed for e-commerce platforms. It addresses the low efficiency and lack of standardization in traditional manual customer service by creatively combining Natural Language Processing (NLP) technology with Large Language Models (LLM). 

The system provides a complete, end-to-end technical workflow encompassing three core AI functions:
1. **Automatic Ticket Classification:** Accurately categorizes user queries (e.g., logistics, complaints).
2. **Sentiment Analysis:** Recognizes the emotional state of the user (positive, neutral, negative).
3. **Intelligent Reply Generation:** Generates context-aware, empathetic, and highly accurate customer service dialogues in real-time.

## Technology Stack

The system adopts a lightweight, hierarchical design integrating both frontend and backend operations:

* **Web Framework & UI:** Flask framework with Jinja2 template engine for dynamic rendering.
* **Database:** SQLite3 for lightweight, persistent data tracking and session management.
* **Machine Learning & NLP:** PyTorch and HuggingFace `transformers`.
* **Deep Learning Models:** Fine-tuned `bert-base-chinese` for semantic comprehension, text classification, and sentiment recognition.
* **Large Language Model (LLM):** Locally deployed `Qwen1.5-1.8B`, utilizing custom Prompt engineering and token-streaming for business-specific reply generation.
* **Security:** `werkzeug.security` (PBKDF2-HMAC-SHA256) for password hashing and secure user authentication.

## System Architecture

The technical implementation follows a three-layer architecture:
* **Bottom Layer (Comprehension):** Pre-trained language models capture the core semantic meaning from raw user inputs.
* **Middle Layer (Adaptation):** Fine-tuned BERT models process the data to output precise classification labels and sentiment tags.
* **Upper Layer (Generation):** The local LLM combines the classification results, sentiment tags, and user text via structured prompts to generate highly natural and compliant responses.

## Repository Notes

* **Excluded Files:** Large model weight files (such as `.safetensors` for BERT and Qwen) and local database files (`.sqlite`) have been excluded from this repository due to GitHub's file size limits. 
* **Purpose:** The provided source code demonstrates the complete engineering implementation, routing logic, database interactions, and training/inference pipelines of the project.
