import base64
import io
import os
from base64 import b64decode
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from langchain.schema.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image


def plt_image_base64(img_base64: str) -> None:
    # Base64データをデコードして画像に変換
    image_data = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(image_data))

    # PILイメージをNumPy配列に変換
    image_np = np.array(image)

    # 画像を表示
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def generate_prompt(data: dict) -> list[HumanMessage]:
    prompt_template = f"""
        以下のcontext（テキスト）のみに基づいて質問に答えてください。
        入力画像が質問に対して関連しない場合には、画像は無視してください。

        質問:
        {data["question"]}

        context:
        {data["context"]["texts"]}
        """
    text_message = {"type": "text", "text": prompt_template}

    # 画像がRetrivalで取得された場合には画像を追加,エンコードしてmatplotlibで表示する
    # 画像が複数取得されている場合には、関連性が最も高いものをモデルへの入力とする
    if data["context"]["images"]:
        # plt_image_base64(data["context"]["images"][0])
        image_url = f"data:image/jpeg;base64,{data['context']['images'][0]}"
        image_message = {"type": "image_url", "image_url": {"url": image_url}}
        return [HumanMessage(content=[text_message, image_message])]
    else:
        return [HumanMessage(content=[text_message])]


# 画像とテキストを分割する
def split_data_type(docs: list[str]) -> dict[str, list[str]]:
    base64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            base64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": base64, "texts": text}


def run_llm_model(
    message: list[BaseMessage], model_name: str = "gemini-2.5-flash-lite"
) -> Any:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    response = model.invoke(message)
    return response
