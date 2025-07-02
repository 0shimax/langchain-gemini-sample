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
        self.modoel_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        # https://smith.langchain.com/hub
        self.prompt = hub.pull("rlm/rag-prompt")

        # self.query_embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/gemini-embedding-exp-03-07", task_type="RETRIEVAL_QUERY"
        # )
        self.doc_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            task_type="RETRIEVAL_DOCUMENT",
        )
        self.vector_store = InMemoryVectorStore(self.doc_embeddings)

    def rag_srouce_loader(
        self,
        url: str = "https://tierzine.com/cats/popular-cat-breeds/",
    ):
        loader = WebBaseLoader(
            web_paths=(url,),
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
    agent.rag_srouce_loader()

    graph_builder = StateGraph(State).add_sequence([agent.retrieve, agent.generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = "温和な正確な猫種を教えて"
    result = graph.invoke({"question": question})
    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
