import os

from google import genai
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from google.genai import types
import pymupdf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class KVCache:
    def __init__(
        self, 
        full_text: str, 
        default_model: str = "microsoft/phi-4"
    ):
        self.past_key_values = self._create_kv_cache(full_text, default_model)
        self.tokenizer = AutoTokenizer.from_pretrained(default_model)
        self.model = AutoModelForCausalLM.from_pretrained(default_model)
        self.knowledgebase = self.tokenizer(full_text, return_tensors="pt").input_ids

    def _create_kv_cache(
        self, 
    ):
        with torch.no_grad():
            output = self.model(self.knowledgebase, use_cache=True)
        return output.past_key_values        

    def _answer_from_cache(self, query:str):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                cache_position=torch.tensor([0]) )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_answer(self, query: str):
        try:
            return self._answer_from_cache(self.past_key_values, query)
        except Exception:
            pass 
        
        knowledge_sentences = self.knowledgebase.split("\n")
        for sentence in knowledge_sentences:
            if query.lower() in sentence.lower():
                new_query = f"""
                query: {query}
                context: {sentence}
                """
                return self.answer_from_cache(self.past_key_values, new_query)
        
        return "I'm sorry, but this query is out of scope for the current knowledge base. Try rephrasing or uploading additional documents."


class RagAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)
        # self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        # # https://smith.langchain.com/hub
        # self.prompt = hub.pull("rlm/rag-prompt")

        # self.doc_embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/text-embedding-004",
        #     task_type="RETRIEVAL_DOCUMENT",
        # )
        # self.vector_store = InMemoryVectorStore(self.doc_embeddings)

    def _load_pdf_source(
        self,
        pdf_path: str,
    ):
        # Open the PDF and extract text
        doc = pymupdf.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc])

    def _cache_context(self, full_text: str):
        return self.client.caches.create(
            model=self.model_name,
            config=types.CreateCachedContentConfig(
                ttl="3600s"  ,
                display_name="PDF_CAG",
                system_instruction=(
                    "You are a highly knowledgeable and detail-oriented AI assistant. "
                    "Your task is to carefully read and understand the document provided. "
                    "You should accurately answer users' questions based solely on the information "
                    "in the document without adding extra knowledge. "
                    "Provide concise, clear, and contextually relevant responses while maintaining professionalism."
                ),
                contents=[full_text]
            )
        )
    
    def build_kv_cache(self, pdf_path: str):
        full_text = self._load_pdf_source(pdf_path)
        self.cache = self._cache_context(full_text)
        self.kv_cache = KVCache(full_text)

    def get_answer(self, question: str):
        return self.kv_cache(question)



if __name__ == "__main__":
    agent = RagAgent()
    agent.build_kv_cache("sample-prj/docs/pdf/improve_guideline.pdf")

    query = input("Talk To Your PDFCAGBot:")
    while query.lower() != "exit":
        response = agent.get_answer(query)
        print("🤖:", response)
        query = input("Say Something:")
