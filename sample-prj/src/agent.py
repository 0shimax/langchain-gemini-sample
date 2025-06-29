import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from google.generativeai.types import Tool as GeminiTool # Import Tool from google.generativeai.types

class OnsenAgent:
    def __init__(self, model_name="gemini-2.5-flash-preview-05-20"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
        # Google Search Toolの定義
        self.tools = [
            GeminiTool(
                function_declarations=[
                    {
                        "name": "google_search",
                        "description": "Google Searchを使って、東京都内の天然温泉銭湯施設に関する情報を検索します。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "検索クエリ"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            )
        ]

    def generate_response(self, user_question):
        try:
            # LLMにツールを使わせる
            response = self.llm.invoke(
                [HumanMessage(content=user_question)],
                tools=self.tools
            )

            # ツール呼び出しがある場合
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.function_call.name == "google_search":
                    search_query = tool_call.function_call.args["query"]
                    # ここで実際のGoogle検索を実行するロジックを実装
                    # 例: google-api-python-client を使用
                    from googleapiclient.discovery import build
                    
                    cse_id = os.getenv("GOOGLE_CSE_ID")
                    if not cse_id:
                        raise ValueError("GOOGLE_CSE_ID environment variable not set for Google Search.")

                    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
                    res = service.cse().list(q=search_query, cx=cse_id).execute()
                    search_results = res.get('items', [])
                    
                    # 検索結果をLLMにフィードバックして最終回答を生成
                    context = "\n".join([item['snippet'] for item in search_results if 'snippet' in item])
                    
                    final_response = self.llm.invoke([
                        SystemMessage(content="あなたは東京都内の天然温泉銭湯施設に関するAIエージェントです。以下の検索結果とユーザーの質問に基づいて、正確で役立つ回答を生成してください。"),
                        HumanMessage(content=f"検索結果:\n{context}\n\nユーザーの質問: {user_question}")
                    ]).content
                    return final_response
                else:
                    return "不明なツールが呼び出されました。"
            else:
                # ツール呼び出しがない場合、直接LLMの回答を返す
                return response.content
        except Exception as e:
            return f"エージェントの実行中にエラーが発生しました: {e}"

if __name__ == "__main__":
    # 環境変数の設定例 (実際にはシェルで設定)
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
    # os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID"

    agent = OnsenAgent()
    print("東京都内天然温泉銭湯施設AIエージェントへようこそ！")
    print("質問を入力してください（終了するには 'exit' と入力）。")

    while True:
        question = input("あなた: ")
        if question.lower() == 'exit':
            break
        
        response = agent.generate_response(question)
        print(f"エージェント: {response}")