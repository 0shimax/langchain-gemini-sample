import os

from google import genai
from langchain_core.documents import Document
from PIL import Image
from typing_extensions import TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class RagAgent:
    # https://qiita.com/birdwatcher/items/b5cc66ce59095dee5625
    # https://recruit.gmo.jp/engineer/jisedai/blog/gemini-multimodal-rag-hands-on/
    # https://note.com/npaka/n/n7a3012b11a95
    # https://qiita.com/birdwatcher/items/b5cc66ce59095dee5625
    # https://ai.google.dev/gemini-api/docs/migrate?hl=ja
    # https://ai.google.dev/gemini-api/docs/embeddings?hl=ja
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)
        # self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk?hl=ja#embeddings
    def embed_content(
        self,
        content: Image.Image,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 1566,  # 3072,
        title: str | None = None,
    ):
        """
        Embed content using Gemini API.
        """
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=content,
            config=genai.types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
                title=title,
            ),
        )
        return response.embeddings


if __name__ == "__main__":
    rag_agent = RagAgent()

    im = Image.open("sample-prj/docs/image/gennba_cat.jpg")
    embeddings = rag_agent.embed_content(content=im, title="Gennba Cat")
    print(embeddings)
    # print(embeddings[0].values)  # Access the embedding values
    # print(embeddings[0].values.shape)  # Check the shape of the embedding
