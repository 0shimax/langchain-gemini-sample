import base64
import io
from base64 import b64decode
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from langchain.chat_models import ChatVertexAI
from langchain.schema.messages import BaseMessage, HumanMessage
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
        以下のcontext（テキストと表）のみに基づいて質問に答えてください。
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
        plt_image_base64(data["context"]["images"][0])
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


# 画像がない場合にはgemini-proを選択する
def model_selection(message: list[BaseMessage]) -> Any:
    if len(message[0].content) == 1:
        answer_generation_model = "gemini-pro"
    else:
        answer_generation_model = "gemini-pro-vision"

    model = ChatVertexAI(model_name=answer_generation_model)
    response = model(message)
    return response
