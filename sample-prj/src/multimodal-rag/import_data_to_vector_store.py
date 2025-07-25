import uuid
from typing import Any

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Helper function to add documents to the vectorstore and docstore
def _add_documents(id_key, retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents, strict=False)))


def import_data_to_vector_store(
    texts_dict: dict[str, list[Any]],
    # tables_dict: dict[str, list[Any]],
    images_dict: dict[str, list[Any]],
    embedding_model_name: str = "models/text-embedding-004",
) -> MultiVectorRetriever:
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        task_type="RETRIEVAL_DOCUMENT",
    )
    vectorstore = Chroma(
        collection_name="gemini-pro-multi-rag",
        embedding_function=embedding_function,
    )

    # 元の文章を保存するためのストレージ
    store = InMemoryStore()
    id_key = "doc_id"

    # Retrieverの作成
    multivector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 6},
    )

    # テキストデータをembedding、vectorstoreに格納する
    _add_documents(
        id_key,
        multivector_retriever,
        texts_dict["texts_list"],
        texts_dict["texts_list"],
    )
    print("### Text Data Stored! ###")

    # 画像データの説明をembedding、vectorstoreに格納する
    _add_documents(
        id_key,
        multivector_retriever,
        images_dict["image_summaries"],
        images_dict["image_list"],
    )
    print("### Image Data Stored! ###")

    return multivector_retriever


if __name__ == "__main__":
    multivector_retriever = import_data_to_vector_store(
        texts_dict=texts_dict,
        images_dict=images_dict,
    )
