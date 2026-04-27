import os
from openai import OpenAI

# DeepSeek 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"  # 也可以用 deepseek-coder

# 读取API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到环境变量 DEEPSEEK_API_KEY，请先配置！")

# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=DEEPSEEK_BASE_URL,
)

# 调用对话接口（问题可以保持不变）
completion = client.chat.completions.create(
    model=DEEPSEEK_CHAT_MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是自然语言处理课程助教，回答要准确、简洁。",
        },
        {
            "role": "user",
            "content": "请用三句话解释什么是自然语言处理。",
        },
    ],
    temperature=0.3,
)

answer = completion.choices[0].message.content
print("DeepSeek回答：")
print(answer)