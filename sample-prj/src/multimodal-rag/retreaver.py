from import_data_to_vector_store import import_data_to_vector_store
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from rag_functions import generate_prompt, run_llm_model, split_data_type
from summarize_images_with_gemini import summarize_images_with_gemini


def generate_retriever(image_dir_path: dict, text_dict: dict) -> MultiVectorRetriever:
    images_dict = summarize_images_with_gemini(image_dir_path)
    return import_data_to_vector_store(texts_dict=text_dict, images_dict=images_dict)


# Chainを作成、実行する
def multimodal_rag(retriever: MultiVectorRetriever, question: str) -> str:
    chain = (
        {
            "context": retriever | RunnableLambda(split_data_type),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(generate_prompt)
        | RunnableLambda(run_llm_model)
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer


if __name__ == "__main__":
    image_dir_path = "sample-prj/docs/image"
    text_file_path = "sample-prj/src/data/text/MAFF.md"

    texts_dict = {}
    with open(text_file_path, encoding="UTF-8") as f:
        text = f.read()
        texts_dict["texts_list"] = [text]
    retriver = generate_retriever(image_dir_path, texts_dict)

    # question = "農水省が安全な作業をするために行っている施策について教えてください"
    question = "水産庁についてなにか言及はありますか？"
    answer = multimodal_rag(
        retriver,
        question,
    )

    print(answer)
