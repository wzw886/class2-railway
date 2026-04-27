from openai import OpenAI

# 阿里云百炼兼容模式配置
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 阿里云 Embedding 模型名称
QWEN_EMBEDDING_MODEL = "text-embedding-v1"

# 你的API Key
api_key = "sk-4743e8bbf36c487e97c7aa36c8deba10"

# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

# 要向量化的文本（可以同时传多个）
texts = [
    "自然语言处理是人工智能的重要分支",
    "大语言模型可以理解和生成人类语言",
    "向量数据库可以存储和检索文本向量"
]

# 调用 Embedding 接口
response = client.embeddings.create(
    model=QWEN_EMBEDDING_MODEL,
    input=texts,
)

# 提取向量结果
embeddings = [item.embedding for item in response.data]

# 打印基础信息
print(f"✅ 成功生成 {len(embeddings)} 个文本向量")
print(f"每个向量的维度：{len(embeddings[0])}")
print("\n第一个向量的前10个值示例：")
print(embeddings[0][:10])