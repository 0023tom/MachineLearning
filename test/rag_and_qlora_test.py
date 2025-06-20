from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class Phi2RAGChatbot:
    def __init__(self, pdf_path, peft_model_path="./qlora_finetuned_model"):
        print("Initializing RAG Phi-2 Chatbot...\n")
        
        # Create offload directory if it doesn't exist
        os.makedirs("./offload", exist_ok=True)
        
        # Load and process PDF document (unchanged RAG implementation)
        print("Loading PDF and building knowledge base...")
        self.raw_text = self._load_pdf(pdf_path)
        self.db = self._build_vectorstore([self.raw_text])
        
        # Load the model with memory optimizations (QLoRA remains unchanged)
        print("Loading Phi-2 model (this may take a few minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Base model loading with offloading support
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder="./offload",  # Added for memory management
            low_cpu_mem_usage=True      # Reduces initial memory spike
        )
        
        # Original QLoRA implementation remains unchanged
        self.model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            offload_folder="./offload"  # Maintains same QLoRA functionality
        )
        
        # Clear memory before proceeding
        torch.cuda.empty_cache()
        self.model.eval()
        print("Chatbot ready!\n")

    # All original RAG methods remain completely unchanged
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
        """Clean and format the generated response"""
        # Original cleaning logic remains identical
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
        """Generate well-formatted answer using RAG (unchanged)"""
        # Original RAG retrieval
        docs = self.db.similarity_search(query, k=4)
        context = "\n".join(f"• {doc.page_content}" for doc in docs)
        
        # Original prompt template
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
        # Original generation logic
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
        
        # Original response processing
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        return self._clean_response(answer)

    # Original chat interface remains unchanged
    def chat(self):
        print("Phi-2 RAG Chatbot 🌟")
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
    chatbot = Phi2RAGChatbot(
        pdf_path="pdf2vault/Short-Info_Sibu_latest.pdf",
        peft_model_path="./qlora_finetuned_model"
    )
    chatbot.chat()