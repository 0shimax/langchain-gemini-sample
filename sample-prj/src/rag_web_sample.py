import os

import bs4
from google import genai
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class RagAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        # https://smith.langchain.com/hub
        self.prompt = hub.pull("rlm/rag-prompt")

        self.doc_embeddings = GoogleGenerativeAIEmbeddings(
            # model="models/gemini-embedding-exp-03-07",
            # model="models/embedding-001",
            model="models/text-embedding-004",
            task_type="RETRIEVAL_DOCUMENT",
        )
        self.vector_store = InMemoryVectorStore(self.doc_embeddings)

    def rag_source_loader(
        self,
        urls: list[str] = [
            # "https://tierzine.com/cats/popular-cat-breeds/",
            "https://tierzine.com/cats/persian-chinchilla-traits/",
            # "https://www.nihonpet.co.jp/cat/mainecoon.html#:~:text=%E7%94%B3%E8%BE%BC%E3%81%BF%E3%81%AF%E3%81%93%E3%81%A1%E3%82%89-,%E3%83%A1%E3%82%A4%E3%83%B3%E3%82%AF%E3%83%BC%E3%83%B3%E3%81%AE%E6%80%A7%E6%A0%BC,%E3%81%99%E3%82%8B%E3%81%93%E3%81%A8%E3%82%92%E5%A5%BD%E3%81%BF%E3%81%BE%E3%81%99%E3%80%82",
            # "https://www.anicom-sompo.co.jp/nekonoshiori/963.html",
            # "https://www.nihonpet.co.jp/cat/ragamuffin.html",
        ],
    ):
        loader = WebBaseLoader(
            web_paths=(*urls,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "post_content",
                        "post_title",
                        "post_header",
                    )
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n", "\n", "。"], chunk_size=100, chunk_overlap=50
        )
        all_splits = text_splitter.split_documents(docs)
        _ = self.vector_store.add_documents(documents=all_splits)

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}


if __name__ == "__main__":
    agent = RagAgent()
    agent.rag_source_loader()

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", agent.retrieve)
    graph_builder.add_node("generate", agent.generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")    
    graph = graph_builder.compile()

    question = "チンチラってどんな猫？"
    result = graph.invoke({"question": question})
    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
