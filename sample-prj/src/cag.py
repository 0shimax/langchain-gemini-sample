import os

import pymupdf
import torch
from google import genai
from google.genai import types
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class KVCache:
    def __init__(self, full_text: str, default_model: str = "microsoft/phi-4"):
        self.tokenizer = AutoTokenizer.from_pretrained(default_model)
        self.model = AutoModelForCausalLM.from_pretrained(default_model)
        self.knowledgebase = self.tokenizer(full_text, return_tensors="pt").input_ids
        self.past_key_values = self._create_kv_cache()

    def _create_kv_cache(self):
        with torch.no_grad():
            output = self.model(self.knowledgebase, use_cache=True)
        return output.past_key_values

    def _answer_from_cache(self, query: str):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        with torch.no_grad():
            # output = self.model.generate(input_ids, cache_position=torch.tensor([0]))
            output = self.model.generate(
                input_ids,
                past_key_values=self.past_key_values,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_answer(self, query: str):
        try:
            return self._answer_from_cache(query)
        except Exception:
            pass

        knowledge_sentences = self.knowledgebase.split("\n")
        for sentence in knowledge_sentences:
            if query.lower() in sentence.lower():
                new_query = f"""
                query: {query}
                context: {sentence}
                """
                return self.answer_from_cache(new_query)

        return "I'm sorry, but this query is out of scope for the current knowledge base. Try rephrasing or uploading additional documents."


class ChatAgent:
    # def __init__(self, model_name: str = "gemini-2.5-flash"):
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)
        # self.chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def _build_chat_config(self, cache):
        return self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(cached_content=cache.name),
        )

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
                ttl="3600s",
                display_name="PDF_CAG",
                system_instruction=(
                    "You are a highly knowledgeable and detail-oriented AI assistant. "
                    "Your task is to carefully read and understand the document provided. "
                    "You should accurately answer users' questions based solely on the information "
                    "in the document without adding extra knowledge. "
                    "Provide concise, clear, and contextually relevant responses while maintaining professionalism."
                ),
                contents=[full_text],
            ),
        )

    def prepare_chat(self, pdf_path: str):
        full_text = self._load_pdf_source(pdf_path)
        self.kv_cache = KVCache(full_text)

        cache = self._cache_context()
        self.chat = self._build_chat_config(cache)

    def get_answer(self, question: str):
        return self.kv_cache.get_answer(question)


if __name__ == "__main__":
    agent = ChatAgent()
    agent.prepare_chat("sample-prj/docs/pdf/improve_guideline.pdf")

    query = input("Talk To Your PDFCAGBot:")
    while query.lower() != "exit":
        response = agent.chat.send_message(query)
        print("🤖:", response)
        query = input("Say Something:")
