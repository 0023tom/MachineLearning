from flask import Flask, request, jsonify
from api.chatbot import Phi2RAGChatbot
import json

app = Flask(__name__)

# Load chatbot once at startup
chatbot = Phi2RAGChatbot(
    pdf_path="pdf2vault/07-GUIDE-SIBU-CENTRAL-v2.pdf",
    peft_model_path="./qlora_finetuned_model"
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request."}), 400

    try:
        question = data["question"]
        print(f"Received question: {question}")
        answer = chatbot.generate_answer(question)
        answer = chatbot.generate_answer(question)
        return app.response_class(
            response=json.dumps({"answer": answer}, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Phi-2 RAG Chatbot API is running!", 200

if __name__ == "__main__":
    app.run(debug=True)