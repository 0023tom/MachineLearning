from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import re
from classifier.filter import TopicFilter
from conversation.memory import ConversationMemory
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class Phi2RAGChatbot:
    def __init__(self, pdf_path, peft_model_path="./qlora_finetuned_model"):
        print("Initializing RAG Phi-2 Chatbot...\n")

        # Load and process PDF document (unchanged RAG implementation)
        print("Loading PDF and building knowledge base...")
        self.raw_text = self._load_pdf(pdf_path)
        self.db = self._build_vectorstore([self.raw_text])
        
        # Load the model with memory optimizations (QLoRA remains unchanged)
        print("Loading Phi-2 model (this may take a few minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Base model loading with offloading support
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        
        # Original QLoRA implementation remains unchanged
        self.model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            use_safetensors=True
        )
        
        # Clear memory before proceeding
        torch.cuda.empty_cache()
        self.model.eval()

        # Init memory and filter
        self.memory = ConversationMemory()
        self.filter = TopicFilter()
        print("Chatbot ready!\n")

    # All original RAG methods remain completely unchanged
    def _load_pdf(self, file_path: str):
        print("Falling back to OCR mode...")

        images = convert_from_path(file_path)
        texts = []

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                print(f"OCR success: Page {i+1}")
                texts.append(text)
            else:
                print(f"OCR failed: Page {i+1}")

        full_text = "\n".join(texts).strip()
        if not full_text:
            raise ValueError("OCR could not extract any text from the PDF.")
        
        # Remove numbered bullet points like "1)", "2)", etc.
        full_text = re.sub(r'\b\d+\)', '', full_text)

        return full_text


    def _build_vectorstore(self, raw_text_list):
        # Check if raw_text_list is empty or contains only whitespace
        if not raw_text_list or all(not text.strip() for text in raw_text_list):
            raise ValueError("No content provided to build the vector store.")

        # Initialize the text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; "]
        )

        # Split the text into documents
        split_docs = splitter.create_documents(raw_text_list)

        # Check if text splitting worked
        if not split_docs:
            raise ValueError("❌ Text splitting failed. No chunks were created.")

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"Loaded text length: {len(raw_text_list[0])}")
        print(f"Chunks created: {len(split_docs)}")

        # Embed and build vectorstore
        return FAISS.from_documents(split_docs, embedding_model)

    def _clean_response(self, text):
        """Clean and truncate the generated response to 2 sentences max."""
        # Strip out any prompt tags
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'###.*?:', '', text)
        
        # Normalize spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first word
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Truncate to first 2 sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        trimmed = ' '.join(sentences[:2])

        return trimmed

    def generate_answer(self, query):
        # Reject unrelated topics
        if not self.filter.is_relevant(query):
            return "Sorry, I can only answer questions related to Sibu and nearby topics."
        # Step 1: Update memory
        self.memory.append_user_input(query)

        # Step 2: Retrieve top docs from vectorstore
        docs = self.db.similarity_search(query, k=4)
        context = "\n".join(f"• {doc.page_content}" for doc in docs)

        if not context.strip():
            return "Sorry, I could not find relevant information about your question."

        # Step 3: Create full prompt
        prompt = f"""### Instruction:
You are a helpful and honest assistant for Sibu-related questions. Use only the provided context. Do not make up information.

### Context:
{context}

### Conversation History:
{self.memory.format_history}

### Question:
{query}

### Rules:
- Only answer using facts from the context above
- Never make up locations or facts
- If the context does not contain the answer, reply: "Sorry, I could not find the answer in my current knowledge."

### Answer:
"""

        # Step 4: Generate response
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.2,
                top_p=0.8,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        final = self._clean_response(answer)

        # Step 5: Update memory with assistant response
        self.memory.append_bot_response(final)
        return final

    # Original chat interface remains unchanged
    def chat(self):
        print("Phi-2 RAG Chatbot")
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