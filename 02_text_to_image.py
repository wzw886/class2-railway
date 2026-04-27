import numpy as np
from openai import OpenAI

# ---------------------- 配置部分（仅修改这里的API Key） ----------------------
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_EMBEDDING_MODEL = "text-embedding-v4"  # 题目要求的模型
api_key = "sk-4743e8bbf36c487e97c7aa36c8deba10"  # 你的阿里云API Key
# ---------------------------------------------------------------------------

# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

# 题目给出的四句话
texts = [
    "我喜欢自然语言处理，尤其是大语言模型。",
    "大模型可以完成文本生成、摘要和问答任务。",
    "今天学校食堂的红烧肉很好吃。",
    "语义向量可以用来计算两个句子的相似度。",
]

# 1. 获取每句话的向量
def get_embedding(text):
    """调用阿里云Embedding模型获取文本向量"""
    response = client.embeddings.create(
        model=QWEN_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(text) for text in texts]

# 2. 题目给出的余弦相似度计算函数
def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

# 3. 计算四句话两两之间的相似度
print("===== 四句话两两相似度 =====")
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"句子{i+1} vs 句子{j+1}：相似度 = {sim:.4f}")
        print(f"文本1：{texts[i]}")
        print(f"文本2：{texts[j]}\n")

# 4. 查找与目标句子语义最相似的句子
target_sentence = "语义向量有哪些作用"
target_embedding = get_embedding(target_sentence)

max_similarity = -1
most_similar_idx = 0

for idx, emb in enumerate(embeddings):
    sim = cosine_similarity(target_embedding, emb)
    if sim > max_similarity:
        max_similarity = sim
        most_similar_idx = idx

print("===== 与目标句子最相似的结果 =====")
print(f"目标句子：{target_sentence}")
print(f"最相似的句子：{texts[most_similar_idx]}")
print(f"相似度：{max_similarity:.4f}")