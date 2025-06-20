# Phi-2 RAG Chatbot for Sibu

This is a Retrieval-Augmented Generation (RAG) chatbot powered by the [Phi-2](https://huggingface.co/microsoft/phi-2) language model fine-tuned using QLoRA. The system is designed to answer questions specifically related to **Sibu, Sarawak**, using a local tourism PDF as its knowledge base.

---

## How to Use

### Step 1: Test Base Model and RAG Setup

Ensure FAISS, vector store, and PDF OCR are working with the base model.

```bash
python3 test/rag_test.py
```

### Step 2: Train Phi-2 with QLoRA

Fine-tune microsoft/phi-2 using your custom dataset.

```bash
python3 test/rag_test.py
```
This uses 4-bit quantization and trains q_proj and v_proj using LoRA.

### Step 3: ✅ Test RAG with QLoRA Adapter
Load the fine-tuned model and test it on relevant questions.

```bash
python3 test/rag_and_qlora_test.py
```

### Step 4: 🧩 Run Chatbot as REST API
Start the Flask API server.

```bash
python3 api/app.py #or python3 -m api.app
```
Then send a test request:

```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Where is Tua Pek Kong temple in Sibu?"}'
```

## Features
- Context-aware, retrieval-grounded chatbot
- Offline operation, no external API calls
- Domain-restricted to Sibu via topic filter
- OCR fallback for PDF text extraction
- Trained and inferenced using 4-bit QLoRA on 4GB VRAM GPU

## References
- microsoft/phi-2 — Base language model
- Sentence Transformers — for vector embeddings
- LangChain + FAISS — document vector storage
- PyTesseract + pdf2image — OCR for scanned PDFs
- HuggingFace QLoRA + PEFT — efficient fine-tuning

## Requirements
Install dependencies:

```bash
pip3 install -r requirements.txt
```

Ensure you have installed:
- CUDA 12.1 (compatible with your GPU).
- poppler-utils (for pdf2image)
- tesseract-ocr (for OCR fallback)
