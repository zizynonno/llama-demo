
from llama_cpp import Llama

class LlamaAdapter():
    def __init__(self,user_content):
        model_path = "./llama.cpp/models/ELYZA-japanese-Llama-2-13b-fast-instruct-q8_0.gguf"
        self.llm = Llama(model_path, chat_format="chatml-function-calling")
        self.output = self.llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "あなたはツアーコンダクターです。",
            },
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "text",
            "schema": {
                "type": "text",
            },
        },
        max_tokens=4000,
        temperature=0.7,
        top_p=0.3,
        top_k=40,
        stream=False,
    )

    def get_response(self):
        return self.output['choices'][0]['message']['content']

def ask(user_content):
    adapter = LlamaAdapter(user_content)
    response = adapter.get_response()
    return response

if __name__ == '__main__':
    user_content = """
    著名な世界遺産を10つ挙げてください。その時なぜその世界遺産がおすすめなのかを記載してください。
    """
    model_output1 = ask(user_content)
    print(model_output1)
