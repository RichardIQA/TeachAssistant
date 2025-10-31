# -*- coding: utf-8 -*-
import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
# import docx
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import networkx as nx
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg'，必须在 import pyplot 之前

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
from typing import List, Tuple
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# =======================
# 1. 配置模型与路径
# =======================

# API Key 和模型配置
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


# =======================
# 2. 加载并分割文档
# =======================

# def load_documents():
#     """加载 data/ 目录下的所有 .docx 文件"""
#     # loader = DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader, loader_kwargs={"autodetect_encoding": "True"})
#     loader = Docx2txtLoader()
#     docs = loader.load()
#     return docs
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


def split_documents(documents):
    """分割文档"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


# =======================
# 3. 构建向量数据库（RAG）
# =======================

def get_or_create_vectorstore():
    if os.path.exists(CHROMA_PATH):
        print("加载已有的向量数据库...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
    else:
        print("正在创建向量数据库...")
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH
        )
        vectorstore.persist()
        print(f"已将 {len(chunks)} 个文本块存入向量库。")
    return vectorstore


# =======================
# 4. 查询重写（Query Rewriting）
# =======================

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


# =======================
# 5. 检索重排（Re-ranking via LLM）
# =======================

def create_compression_retriever(vectorstore):
    """创建带重排序的压缩检索器"""
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    return compression_retriever


# =======================
# 6. 知识图谱构建（轻量级）
# =======================

# def extract_triples(text: str) -> List[Tuple[str, str, str]]:
#     """
#     从文本中简单提取三元组（实体-关系-实体）
#     实际项目中可用 NLP 模型（如 spaCy、LTP）提升效果
#     """
#     # 简化规则：匹配 "A 是 B"、"A 包括 B"、"A 导致 B" 等模式
#     patterns = [
#         (r"(.+?)是(.+?)", "是"),
#         (r"(.+?)包括(.+?)", "包括"),
#         (r"(.+?)属于(.+?)", "属于"),
#         (r"(.+?)导致(.+?)", "导致"),
#         (r"(.+?)影响(.+?)", "影响"),
#         (r"(.+?)组成(.+?)", "组成"),
#     ]
#     triples = []
#     for pattern, rel in patterns:
#         matches = re.findall(pattern, text)
#         for m in matches:
#             if isinstance(m, tuple):
#                 subj, obj = m[0].strip(), m[1].strip()
#             else:
#                 subj, obj = "", m.strip()
#             if len(subj) > 1 and len(obj) > 1:
#                 triples.append((subj, rel, obj))
#     return triples

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


# def build_knowledge_graph(chunks) -> nx.Graph:
#     """从文档块构建知识图谱"""
#     G = nx.Graph()
#     for chunk in chunks:
#         text = chunk.page_content
#         triples = extract_triples(text)
#         for subj, rel, obj in triples:
#             G.add_node(subj)
#             G.add_node(obj)
#             G.add_edge(subj, obj, relation=rel)
#     print(f"知识图谱构建完成：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
#     return G


def build_knowledge_graph(chunks) -> nx.Graph:
    G = nx.Graph()
    for chunk in chunks:
        text = chunk.page_content
        triples = extract_triples(text)
        for subj, rel, obj in triples:
            G.add_node(subj)
            G.add_node(obj)
            # 合并相同关系
            if G.has_edge(subj, obj):
                # 如果已有边，合并关系（如“是, 包括”）
                existing_rel = G[subj][obj].get("relation", "")
                if rel not in existing_rel:
                    G[subj][obj]["relation"] = existing_rel + ", " + rel
            else:
                G.add_edge(subj, obj, relation=rel)
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
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(subgraph, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("教学知识图谱（Top 20 节点）")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =======================
# 7. RAG 问答链
# =======================

def create_rag_chain(retriever):
    """创建 RAG 链"""
    prompt_template = """
你是一个智能教学反馈助手。请根据以下上下文回答问题，并完成两个任务：

1. **提炼重点知识点**（列出 3-5 条，每条不超过 50 字）
2. **生成 3 道练习题**（选择题或简答题，附带答案）

上下文信息：
{context}

用户问题：{question}

请按以下格式输出：
---
### 重点知识点：
1. ...
2. ...

### 练习题：
1. 问题...
   答案：...
2. ...
---
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


# =======================
# 8. 主程序入口
# =======================

def main():
    print("📚 智能教学反馈智能体启动中...")
    
    # 加载文档
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"共加载 {len(docs)} 个文档，分割为 {len(chunks)} 个文本块。")
    
    # 向量库
    vectorstore = get_or_create_vectorstore()
    
    # 压缩检索器（带重排序）
    retriever = create_compression_retriever(vectorstore)
    
    # 构建知识图谱
    print("正在构建知识图谱...")
    kg = build_knowledge_graph(chunks)
    visualize_kg(kg)
    
    # 创建 RAG 链
    rag_chain = create_rag_chain(retriever)
    
    # 交互循环
    print("\n🔍 智能教学助手已准备就绪！输入问题获取知识点和练习题（输入 'quit' 退出，'kg' 查看知识图谱）：")
    
    while True:
        query = input("\n📥 你的问题：").strip()
        if query.lower() == "quit":
            print("👋 再见！")
            break
        if query.lower() == "kg":
            visualize_kg(kg)
            continue
        if not query:
            continue
        
        print("🔄 正在处理...")
        
        # 查询重写
        rewritten_query = rewrite_query(query)
        print(f"📝 重写查询：{rewritten_query}")
        
        # RAG 回答
        try:
            response = rag_chain.invoke(rewritten_query)
            print("\n✅ 智能反馈：")
            for chunk in response:
                print(chunk, end='', flush=True)
        except Exception as e:
            print(f"❌ 生成失败：{e}")


if __name__ == "__main__":
    main()
