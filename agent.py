# agent.py - 更新部分
import os
import re

import networkx as nx
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StreamingStdOutCallbackHandler
from typing import Any, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，必须在 import pyplot 之前

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# 自定义流式回调处理器（用于返回前端）
ZHIPU_API_KEY = "0f4ae0b90dff44389836ecf634297560.c1eOL2jdMckW1bfO"

# 初始化 LLM 和 Embedding
llm = ChatZhipuAI(
    temperature=0.95,
    model="glm-4",
    api_key=ZHIPU_API_KEY,
)

embedding_model = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=ZHIPU_API_KEY
)

# 文档路径
DATA_DIR = "data"  # 请确保此文件夹存在并包含 .docx 文件
CHROMA_PATH = "chroma_db"


def rewrite_query(original_query: str) -> str:
    """使用 LLM 重写用户查询，增强语义"""
    rewrite_prompt = PromptTemplate.from_template(
        """
    你是一个教学辅助查询重写助手，负责帮助教师或学生将原始问题优化为更适合课堂知识检索、习题讲评和学习反思的提问形式。

    请基于以下原则重写问题：
    1. 保持原意，不改变提问的核心意图；
    2. 提升表达的清晰度与学术性，适合用于课件、教案、错题本或学习资料的检索；
    3. 显式突出以下一个或多个维度（如适用）：
       - 【重点知识】：涉及的核心概念、课程标准要求的关键能力；
       - 【易错点】：学生常犯的误解、混淆概念或逻辑漏洞；
       - 【做题技巧】：解题的典型方法、策略或思维路径（如分类讨论、数形结合等）；
       - 【规律总结】：可推广的结论、模型或解题通法（如“滑动变阻器分压特性”“三角函数周期规律”）；
    4. 适当扩展关键词或学科术语，增强与教学资源的匹配度；
    5. 若问题模糊，推测其可能指向的知识模块，并合理具象化为典型问题形态；
    6. 避免口语化表达，使用规范的学科语言和教学表述。

    原问题：{query}

    请输出重写后的问题，仅输出重写结果，不要添加解释：
            """
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = rewrite_chain.invoke({"query": original_query})
    return rewritten.strip()


class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_token):
        self.on_token = on_token
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.on_token(token)


# 全局变量，用于存储组件（避免重复初始化）
_cached = {}


def init_agent():
    if "initialized" in _cached:
        return _cached
    
    print("🧠 正在初始化智能教学系统...")
    
    # ===== 原有初始化逻辑 =====
    from langchain_community.document_loaders import DirectoryLoader
    
    embedding_model = ZhipuAIEmbeddings(
        model="embedding-3",
        api_key="0f4ae0b90dff44389836ecf634297560.c1eOL2jdMckW1bfO"
    )
    
    DATA_DIR = "data"
    CHROMA_PATH = "chroma_db"
    
    def load_documents():
        """加载 data/ 目录下的所有 .docx 文件"""
        docs = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(".docx"):
                    file_path = os.path.join(root, file)
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())
        return docs
    
    def split_documents(docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_documents(docs)
    
    docs = load_documents()
    chunks = split_documents(docs)
    
    if not os.path.exists(CHROMA_PATH):
        vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH)
        vectorstore.persist()
    else:
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # LLM 支持流式输出
    llm = ChatZhipuAI(
        temperature=0.95,
        model="glm-4",
        api_key="0f4ae0b90dff44389836ecf634297560.c1eOL2jdMckW1bfO",
        streaming=True,
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()])  # 控制台流式
    )
    
    # 提取上下文函数
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    # 不在此处绑定完整 chain，留待流式调用
    _cached.update({
        "retriever": retriever,
        "llm": llm,
        "chunks": chunks,
        "format_docs": format_docs,
        "initialized": True
    })
    return _cached


import jieba  # 用于中文分词，提升实体边界识别


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    优化版三元组抽取（轻量规则 + 分词 + 常见句式）
    """
    if not text.strip():
        return []
    
    # 分句
    sentences = re.split(r"[。！？;\n]", text)
    triples = []
    
    # 扩展模式：支持更多句式，使用捕获组
    patterns = [
        # 核心模式
        (r"(.+?)是(.+?)[的]*(?:概念|系统|方法|技术|模型|算法|原理)", "是"),
        (r"(.+?)是(.+?)", "是"),  # 通用是
        (r"(.+?)包括(.+?)(?:、(.+?))*", "包括"),
        (r"(.+?)包含(.+?)(?:、(.+?))*", "包含"),
        (r"(.+?)属于(.+?)", "属于"),
        (r"(.+?)导致(.+?)", "导致"),
        (r"(.+?)影响(.+?)", "影响"),
        (r"(.+?)组成(.+?)", "组成"),
        (r"(.+?)由(.+?)组成", "由...组成"),  # 反向
        (r"(.+?)用于(.+?)", "用于"),
        (r"(.+?)实现(.+?)", "实现"),
        (r"(.+?)基于(.+?)", "基于"),
        (r"(.+?)分为(.+?)(?:、(.+?))*", "分为"),
    ]
    
    for sent in sentences:
        sent = re.sub(r"[\s（）\(\)]+", "", sent.strip())  # 去空格和括号
        if len(sent) < 4:
            continue
        
        for pattern, rel in patterns:
            matches = re.findall(pattern, sent)
            for m in matches:
                if isinstance(m, tuple):
                    # 处理多捕获组，如“包括 A、B、C”
                    parts = [part.strip() for part in m if part.strip()]
                    if len(parts) < 2:
                        continue
                    subj = parts[0]
                    objects = parts[1:]
                else:
                    subj, obj = "", m.strip()
                    objects = [obj]
                
                # 对每个 object 生成三元组
                for obj in objects:
                    if len(subj) > 1 and len(obj) > 1:
                        # 使用 jieba 简单分词，避免“数学中的基本概念”这种超长实体
                        subj = refine_entity(subj)
                        obj = refine_entity(obj)
                        triples.append((subj, rel, obj))
    
    return triples


def refine_entity(entity: str) -> str:
    """
    简单优化实体：去除冗余词、切分过长实体
    """
    # 常见过滤词
    stop_words = {"的", "一种", "一个", "一类", "所谓", "所谓", "基本", "重要", "主要"}
    words = jieba.lcut(entity)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    if not words:
        return entity.strip()
    # 取核心词（可改进为 TF-IDF 或关键词提取）
    return "".join(words[-2:]) if len(words) >= 2 else words[0]

def build_knowledge_graph(chunks) -> nx.Graph:
    """从文档块构建知识图谱"""
    G = nx.Graph()
    for chunk in chunks:
        text = chunk.page_content
        triples = extract_triples(text)
        for subj, rel, obj in triples:
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, relation=rel)
    print(f"知识图谱构建完成：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    return G


def visualize_kg(G, top_k=20):
    """可视化知识图谱（前 top_k 个重要节点）"""
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
    subgraph_nodes = [node for node, _ in top_nodes]
    subgraph = G.subgraph(subgraph_nodes)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, k=0.5)
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(subgraph, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("教学知识图谱（Top 20 节点）")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 测试（临时）
if __name__ == "__main__":
    init_agent()
    print(rewrite_query("牛顿定律是啥？"))
