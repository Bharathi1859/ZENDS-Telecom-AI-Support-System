# 📡 ZENDS Telecom AI Support System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)


An enterprise-grade AI-powered customer support automation platform built using Transformer models, Retrieval-Augmented Generation (RAG), and an interactive Streamlit dashboard.

### 🚀 Project Overview

The ZENDS Telecom AI Support System is designed to automate and enhance telecom customer service operations using modern Natural Language Processing (NLP) techniques.

The system integrates:

- 🎯 Intent Classification (Transformer-based)

- 😊 Sentiment Analysis

- 🔍 Retrieval-Augmented Generation (RAG)

- 🧠 Vector Similarity Search using ChromaDB

- ✏️ Human-in-the-Loop Response Editing

- 📊 Interactive Analytics Dashboard

- 🌗 Dark / Light Theme Enterprise UI

### 🛠 Tech Stack

Programming Language: Python
Frontend Framework: Streamlit
Deep Learning Framework: PyTorch
NLP Models: HuggingFace Transformers
Embedding Model: SentenceTransformer (all-MiniLM-L6-v2)
Text Generation Model: Google FLAN-T5
Vector Database: ChromaDB
Visualization: Plotly

### 🧠 Key Features

- Automatic intent detection (Technical, Billing, Complaint, Refund, Product Inquiry)

- Customer sentiment analysis (Positive, Neutral, Negative)

- Context-aware response generation using RAG

- Editable AI responses via Support Agent Panel

- Real-time analytics and data visualization

- Modular and scalable architecture

## ⚠️ Model Files Notice

Due to GitHub file size limitations (200MB per file), 
the trained transformer models are not included in this repository.

The models were fine-tuned using Google Colab GPU 
and stored locally for deployment.

To use this project:

1. Train the models using provided notebook (Colab)
2. Save them as:
   - intent_model/
   - sentiment_model/
3. Place them inside the project directory

### (OR)

## 📥 Download Trained Models

Intent Model/sentiment model: [(https://drive.google.com/drive/folders/1-xQ9CJyzk0lUKQNxQnMl2PmDCgYI6-cg?usp=drive_link)]

Download and place inside project folder before running.


### 🔍 Why RAG?

Retrieval-Augmented Generation ensures:

- Reduced hallucination

- Improved factual accuracy

- Context-based responses

- Enterprise-level reliability

### 📊 Analytics Dashboard

The system provides real-time monitoring of:

- Total queries processed

- Intent distribution

- Sentiment distribution

This enables data-driven business insights.

### 🏆 Project Highlights

- Transformer-based NLP implementation

- GPU fine-tuning using Google Colab

- Deployment-ready Streamlit application

- Professional SaaS dashboard design

- Human-AI collaborative system

### 🔮 Future Enhancements

- Cloud deployment (AWS / Azure / GCP)

- Authentication and role management

- Voice-based support integration

- Continuous learning pipeline

- Real-time alert system for negative sentiment

### 📌 Conclusion

This project demonstrates the practical implementation of modern AI techniques in enterprise customer support automation, combining NLP, deep learning, vector databases, and analytics into a unified intelligent system.
