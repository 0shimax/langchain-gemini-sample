import os
import traceback

from google import genai
from google.genai import types
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class OnsenAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.modoel_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # Only run this block for Gemini Developer API
        self.client = genai.Client(api_key=api_key)

        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def serach_web(self, user_question: str):
        # Google Search Toolの定義
        # https://ai.google.dev/gemini-api/docs/migrate?hl=ja
        # https://github.com/googleapis/python-genai
        response = self.client.models.generate_content(
            model=self.modoel_name,
            contents=user_question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_modalities=["TEXT"],
            ),
        )
        return response.candidates[0].content.parts[0].text

    def post_process(self, context: str, user_question: str):
        final_response = self.llm.invoke(
            [
                SystemMessage(
                    content="あなたは東京都内の天然温泉銭湯施設に関するAIエージェントです。以下の検索結果とユーザーの質問に基づいて、正確で役立つ回答を生成してください。"
                ),
                HumanMessage(
                    content=f"検索結果:\n{context}\n\nユーザーの質問: {user_question}"
                ),
            ]
        ).content
        return final_response

    def explore(self, user_question: str):
        try:
            context = self.serach_web(user_question)
            return self.post_process(context, user_question)
        except Exception:
            error_msg = traceback.format_exc()
            msg = f"エージェントの実行中にエラーが発生しました: {error_msg}"
            raise RuntimeError(msg)


if __name__ == "__main__":
    agent = OnsenAgent()
    print("東京都内天然温泉銭湯施設AIエージェントへようこそ！")
    print("質問を入力してください（終了するには 'exit' と入力）。")

    while True:
        question = input("質問: ")
        if question.lower() == "exit":
            break

        response = agent.explore(question)
        print(f"エージェント: {response}")
