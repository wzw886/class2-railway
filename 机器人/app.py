from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
import os

# 初始化FastAPI应用
app = FastAPI()

# 关键：获取当前文件的目录，确保路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 挂载静态文件目录（必须和app.py同级的static文件夹）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 配置阿里云通义千问模型
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"
# 你的API Key
api_key = "DASHSCAPE_API_KEY"

client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str

# 首页路由，正确返回static里的index.html
@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)

# 聊天接口
@app.post("/chat")
async def chat(request: ChatRequest):
    # 调用大模型
    completion = client.chat.completions.create(
        model=QWEN_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个友好的AI助手，回答简洁清晰。"},
            {"role": "user", "content": request.message}
        ],
        temperature=0.7,
    )
    # 返回回答
    return {"reply": completion.choices[0].message.content}