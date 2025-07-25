import base64
import os
from typing import Any

from dotenv import load_dotenv
from google import genai

# from google.genai.types import Part

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")


# 画像ファイルをBase64エンコードされた文字列に変換
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


# Gemini Pro Visionにて画像の説明を行い、説明結果とbase64を返却する
def summarize_images_with_gemini(image_dir: str) -> dict[str, list[Any]]:
    image_base64_list = []
    image_summaries_list = []
    image_summary_prompt = """
    入力された画像の内容を詳細に説明してください。
    基本的には日本語で回答してほしいですが、専門用語や固有名詞を用いて説明をする際には英語のままで構いません。
    はい、いいえなどは省いて説明内容だけを出力して下さい。
    """

    client = genai.Client(api_key=api_key)

    for image_file_name in sorted(os.listdir(image_dir)):
        if image_file_name.endswith(".jpg"):
            image_file_path = os.path.join(image_dir, image_file_name)

            # encodeを行い、base64をリストに格納する
            image_base64 = image_to_base64(image_file_path)
            image_base64_list.append(image_base64)

            with open(image_file_path, "rb") as f:
                image_bytes = f.read()

            # Geminiで画像の説明を行い、結果をリストに格納する
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    image_summary_prompt,
                    genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )

            image_summaries_list.append(response.text.strip())
            images_dict = {
                "image_list": image_base64_list,
                "image_summaries": image_summaries_list,
            }

    return images_dict


if __name__ == "__main__":
    image_dir_path = "sample-prj/docs/image"
    images_dict = summarize_images_with_gemini(image_dir_path)
    print(images_dict)
