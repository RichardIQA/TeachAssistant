# app.py - 支持流式输出

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import asyncio
from starlette.middleware.cors import CORSMiddleware
from agent import init_agent, StreamCallbackHandler, rewrite_query, build_knowledge_graph

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="智能教学反馈系统（流式版）")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化智能体
init_agent()
cached = init_agent()


# 数据模型
class QueryRequest(BaseModel):
    question: str


async def stream_generator(question: str):
    try:
        # 1. 查询重写
        rewritten = rewrite_query(question)
        yield f"data: {json.dumps({'type': 'rewrite', 'text': rewritten})}\n\n"
        
        # 2. 检索
        docs = cached["retriever"].invoke(rewritten)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
                你是一个智能教学助手。请根据以下上下文回答问题，并完成两个任务：
                
                1. **提炼重点知识点**
                2. **生成 3 道练习题**（附答案）
                
                上下文：
                {context}
                
                问题：{rewritten}
                请按格式输出：
                ---
                ### 重点知识点：
                ...
                
                ### 练习题：
                ...
                ---
        """
        
        # 3. 流式生成
        async def on_token(token: str):
            await asyncio.sleep(0)  # 协程让步
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        
        # 使用自定义回调
        callback = StreamCallbackHandler(lambda t: asyncio.create_task(on_token(t)))
        cached["llm"].callbacks = None  # 避免冲突
        cached["llm"].callbacks = [callback]
        
        # 手动模拟流式生成（LangChain 的 invoke 不直接支持异步流，我们用同步生成）
        from langchain_core.messages import HumanMessage
        
        full_response = ""
        for chunk in cached["llm"].stream([HumanMessage(content=prompt)]):
            text = chunk.content
            if text:
                full_response += text
                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            await asyncio.sleep(0.01)
        
        yield f"data: {json.dumps({'type': 'done', 'text': ''})}\n\n"
    
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"


@app.post("/query")
async def query_stream(request: QueryRequest):
    return StreamingResponse(
        stream_generator(request.question.strip()),
        media_type="text/event-stream"
    )


@app.get("/kg")
async def get_kg():
    try:
        # 获取所有文档块（用于构建图谱）
        docs = cached["retriever"].invoke("教学内容")  # 可以用一个通用查询获取全部内容
        G = build_knowledge_graph(docs)  # 调用 agent.py 中的函数
        
        # 转换为前端可用的格式
        nodes = [{"id": node, "label": node} for node in G.nodes()]
        edges = [{"source": edge[0], "target": edge[1], "relation": G[edge[0]][edge[1]]['relation']} for edge in
                 G.edges()]
        
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"❌ 知识图谱构建失败: {e}")
        return {"nodes": [], "edges": []}  # 返回空数据避免崩溃


@app.get("/health")
async def health():
    return {"status": "ok"}


# 获取当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
print("📁 静态目录:", static_dir)
print("📄 文件列表:", os.listdir(static_dir) if os.path.exists(static_dir) else "不存在")
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()
