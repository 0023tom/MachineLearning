from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class Phi2RAGChatbot:
    def __init__(self, pdf_path):
        print("Initializing RAG Phi-2 Chatbot (Base model only)...\n")
        
        os.makedirs("./offload", exist_ok=True)

        # Load and process PDF document
        print("Loading PDF and building knowledge base...")
        self.raw_text = self._load_pdf(pdf_path)
        self.db = self._build_vectorstore([self.raw_text])

        # Load tokenizer and base Phi-2 model (no PEFT)
        print("Loading base Phi-2 model (no fine-tuning)...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder="./offload",
            low_cpu_mem_usage=True
        )

        torch.cuda.empty_cache()
        self.model.eval()
        print("Chatbot ready!\n")

    def _load_pdf(self, file_path: str):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    def _build_vectorstore(self, raw_text_list):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; "]
        )
        split_docs = splitter.create_documents(raw_text_list)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.from_documents(split_docs, embedding_model)

    def _clean_response(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'###.*?:', '', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        return text

    def generate_answer(self, query):
        docs = self.db.similarity_search(query, k=4)
        context = "\n".join(f"• {doc.page_content}" for doc in docs)

        prompt = f"""### Instruction:
Provide a 1-2 sentence answer using ONLY these facts:

### Context:
{context}

### Question:
{query}

### Rules:
- Be extremely concise
- Never exceed 2 sentences
- Skip promotional language
- Only include verified facts

### Answer:
"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        return self._clean_response(answer)

    def chat(self):
        print("Phi-2 RAG Chatbot (Base model only)")
        print("Type 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nChatbot: Thank you! Have a great day!")
                    break

                print("Searching knowledge base...")
                response = self.generate_answer(user_input)
                print(f"\nChatbot: {response}\n")

            except KeyboardInterrupt:
                print("\nChatbot: Session ended.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Chatbot: My apologies, I encountered an issue. Please try again.\n")

if __name__ == "__main__":
    chatbot = Phi2RAGChatbot(pdf_path="pdf2vault/Short-Info_Sibu_latest.pdf")
    chatbot.chat()