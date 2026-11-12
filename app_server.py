import os
import re
import json
import inspect
import logging
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from contextvars import ContextVar
import numpy as np
import httpx
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel

# ========== tiktoken 补丁 ==========
tiktoken_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiktoken_cache")
os.makedirs(tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
import tiktoken

_orig_efm = tiktoken.encoding_for_model
def _custom_efm(m):
    try:
        return _orig_efm(m)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
tiktoken.encoding_for_model = _custom_efm

_orig_ge = tiktoken.get_encoding
def _custom_ge(n):
    try:
        return _orig_ge(n)
    except Exception:
        return _orig_ge("cl100k_base")
tiktoken.get_encoding = _custom_ge

from hirag import HiRAG, QueryParam
from hirag._utils import EmbeddingFunc, compute_args_hash

logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("HiRAG").setLevel(logging.ERROR)

# ==========================================
# 配置：支持 Agnes API 与本地 Ollama (输入 "ollama" 或 "local" 自动切本地 7B)
# ==========================================
WORK_DIR           = "./pdf_work_dir_local"
GRAPHML_PATH       = os.path.join(WORK_DIR, "graph_chunk_entity_relation.graphml")
CHUNKS_JSON_PATH   = os.path.join(WORK_DIR, "kv_store_text_chunks.json")

LLM_BASE_URL       = "https://apihub.agnes-ai.com/v1"
LLM_MODEL_NAME     = "agnes-2.5-flash"

LOCAL_OLLAMA_V1    = "http://localhost:11434/v1"
LOCAL_LLM_MODEL    = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:7b")

EMBED_BASE_URL     = LOCAL_OLLAMA_V1
EMBED_API_KEY      = "ollama"
EMBED_MODEL_NAME   = "bge-m3:latest"
EMBED_DIM          = 1024

_LLM_HTTP_CLIENT   = httpx.AsyncClient(trust_env=True, timeout=3600.0)
_EMBED_HTTP_CLIENT = httpx.AsyncClient(trust_env=False, timeout=120.0)

_EMBED_CLIENT = AsyncOpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
    http_client=_EMBED_HTTP_CLIENT,
)

# 协程上下文变量：传递前端输入的 API Key、当前用户提问与捕获的底层证据链
_CURRENT_API_KEY: ContextVar[str] = ContextVar("_CURRENT_API_KEY", default="")
_CURRENT_USER_QUERY: ContextVar[str] = ContextVar("_CURRENT_USER_QUERY", default="")
_CAPTURED_EVIDENCE: ContextVar[str] = ContextVar("_CAPTURED_EVIDENCE", default="")

_KNOWN_ENTITY_IDS: set[str] = set()
_CHUNKS_STORE: dict[str, dict] = {}
_DOC_TO_CHUNKS: dict[str, list[tuple[int, str, str]]] = {}  # full_doc_id -> [(order, cid, content)]
_DOC_ALIASES: dict[str, str] = {}  # 别名/合同编号 -> full_doc_id

# 排除高频泛化噪音节点，避免整张图到处乱闪
_STOP_ENTITIES = {
    "CONTRACTOR", "THE CONTRACTOR", "MAIN CONTRACTOR", "SUB-CONTRACTOR",
    "SUBCONTRACTOR", "SUB-CONTRACTORS", "EMPLOYER", "WORKS", "THE WORKS",
    "HONG KONG", "GOVERNMENT", "GOVERNMENT OF HONG KONG", "TENDERER",
    "TENDERERS", "SAMPLES", "DRAWINGS", "MATERIAL", "SITE", "PROJECT SITE",
    "CONTRACT", "TENDER", "CLAUSE 2", "SUB-CLAUSE (A)", "SUB-CLAUSE (B)", "SUB-CLAUSE (C)"
}


def load_known_entity_ids_and_chunks():
    """加载图谱节点与文本切片索引，建立【合同编号/样本别名 -> 完整同源切片组】映射"""
    _KNOWN_ENTITY_IDS.clear()
    _CHUNKS_STORE.clear()
    _DOC_TO_CHUNKS.clear()
    _DOC_ALIASES.clear()

    if os.path.exists(CHUNKS_JSON_PATH):
        try:
            with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                _CHUNKS_STORE.update(json.load(f))
            for cid, cdata in _CHUNKS_STORE.items():
                doc_id = cdata.get("full_doc_id") or "unknown_doc"
                order = int(cdata.get("chunk_order_index", 0))
                content = cdata.get("content", "")
                _DOC_TO_CHUNKS.setdefault(doc_id, []).append((order, cid, content))

                src_m = re.search(r"\[DOCUMENT_SOURCE:\s*([^\]]+)\]", content)
                if src_m:
                    fname = src_m.group(1).strip()
                    stem = os.path.splitext(fname)[0].upper()
                    _DOC_ALIASES[fname.upper()] = doc_id
                    _DOC_ALIASES[stem] = doc_id
                    if "CONTRACT_A" in stem:
                        for alias in ("样本A", "样本 A", "合同A", "合同 A", "CONTRACT A", "CONTRACT_A", "主合同"):
                            _DOC_ALIASES[alias] = doc_id
                    if "CONTRACT_B" in stem:
                        for alias in ("样本B", "样本 B", "合同B", "合同 B", "CONTRACT B", "CONTRACT_B", "分包合同"):
                            _DOC_ALIASES[alias] = doc_id

                if "HDIL-2026-MC-01" in content and "DOMESTIC SUBCONTRACT" not in content[:200]:
                    _DOC_ALIASES["HDIL-2026-MC-01"] = doc_id
                if "APEX-2026-SUB-MEP-01" in content:
                    _DOC_ALIASES["APEX-2026-SUB-MEP-01"] = doc_id

            for doc_id in _DOC_TO_CHUNKS:
                _DOC_TO_CHUNKS[doc_id].sort(key=lambda x: x[0])
        except Exception as e:
            print(f"⚠️ 切片索引构建异常: {e}")

    if os.path.exists(GRAPHML_PATH):
        try:
            tree = ET.parse(GRAPHML_PATH)
            for elem in tree.getroot().iter():
                if elem.tag.endswith("node"):
                    raw_id = elem.attrib.get("id", "").strip().strip('"')
                    if raw_id:
                        _KNOWN_ENTITY_IDS.add(raw_id)
        except Exception as e:
            print(f"⚠️ GraphML 节点预扫描异常: {e}")


def build_supplemental_chunks_context(query: str, system_prompt: str) -> str:
    """
    当用户询问特定合同（如样本 A / HDIL-2026-MC-01）或特定条款（如第 4 条）时，
    按切片中部独有指纹判断并补齐全部同源切片（part 1/5 ~ 5/5），彻底解决因 Overlap 重叠导致的 part 2/5 漏检问题。
    """
    if not _DOC_TO_CHUNKS:
        return ""

    target_doc_ids = set()
    q_upper = query.upper()

    # 1. 优先仅根据用户提问(query)锁定目标合同，避免把未提问的其他合同塞入上下文拖慢速度
    for alias, doc_id in _DOC_ALIASES.items():
        if alias in q_upper:
            target_doc_ids.add(doc_id)

    # 2. 仅当用户提问未指明具体合同时，才从已召回的 system_prompt 中检测短合同
    if not target_doc_ids:
        for doc_id, chunk_list in _DOC_TO_CHUNKS.items():
            if len(chunk_list) <= 12:
                for _, cid, content in chunk_list:
                    mid = len(content) // 2
                    unique_probe = content[max(0, mid - 80) : mid + 80].strip()
                    if len(unique_probe) > 30 and unique_probe in system_prompt:
                        target_doc_ids.add(doc_id)
                        break

    clause_nums = re.findall(r"(?:第\s*(\d+(?:\.\d+)?)\s*条|CLAUSE\s+(\d+(?:\.\d+)?))", query, flags=re.I)
    flat_clauses = [c[0] or c[1] for c in clause_nums if (c[0] or c[1])]

    supplement_blocks = []
    for doc_id in target_doc_ids:
        chunk_list = _DOC_TO_CHUNKS.get(doc_id, [])
        if not chunk_list:
            continue

        if len(chunk_list) <= 12:
            for order, cid, content in chunk_list:
                # 关键修复：取切片正中间 160 字符作为独有指纹判重，防止 part 2/5 因开头与 part 1/5 重叠而被误杀
                mid = len(content) // 2
                unique_probe = content[max(0, mid - 80) : mid + 80].strip()
                if not unique_probe or unique_probe not in system_prompt:
                    supplement_blocks.append(
                        f"<!-- Supplemental Sibling Chunk ({cid}, part {order + 1}/{len(chunk_list)}) -->\n{content}"
                    )
        elif flat_clauses:
            for order, cid, content in chunk_list:
                mid = len(content) // 2
                unique_probe = content[max(0, mid - 80) : mid + 80].strip()
                for cl_num in flat_clauses:
                    pattern = rf"(?m)(?:^\s*(?:\[[^\]]+\]\s*)?{re.escape(cl_num)}\b|CLAUSE\s+{re.escape(cl_num)}\b)"
                    if re.search(pattern, content, flags=re.I) and unique_probe not in system_prompt:
                        supplement_blocks.append(
                            f"<!-- Supplemental Clause {cl_num} Chunk ({cid}) -->\n{content}"
                        )
                        break

    if not supplement_blocks:
        return ""

    return (
        "-----Supplemental Complete Contract Chunks (Auto-Completed by Document ID)-----\n"
        + "\n\n".join(supplement_blocks)
    )


def build_safe_query_param() -> QueryParam:
    """精简检索规模，砍掉 gccbwt.doc 的长尾噪音，将单次查询速度提升 3 倍以上"""
    valid_keys = set(inspect.signature(QueryParam.__init__).parameters.keys())
    desired = {
        "mode": "hi",
        "top_k": 12,
        "top_m": 5,
        "max_token_for_text_unit": 3500,
        "max_token_for_local_context": 3500,
        "max_token_for_community_report": 2000,
    }
    filtered = {k: v for k, v in desired.items() if k in valid_keys}
    return QueryParam(**filtered)


async def llm_model_if_cache(
    prompt: str, system_prompt: str = "", history_messages: list = None, **kwargs
) -> str:
    if history_messages is None:
        history_messages = []
    hashing_kv = kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    api_key = _CURRENT_API_KEY.get().strip() or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("未提供有效的 API Key，请在页面输入 API Key（或输入 ollama 使用本地模型）。")

    use_local_ollama = api_key.lower() in ("ollama", "local", "qwen")
    active_model = LOCAL_LLM_MODEL if use_local_ollama else LLM_MODEL_NAME
    active_base_url = LOCAL_OLLAMA_V1 if use_local_ollama else LLM_BASE_URL
    active_http_client = _EMBED_HTTP_CLIENT if use_local_ollama else _LLM_HTTP_CLIENT

    is_rag_response = (
        len(system_prompt) > 500
        or "---Role---" in system_prompt
        or "-----Entities-----" in system_prompt
    )

    effective_system_prompt = system_prompt
    if is_rag_response:
        user_q = _CURRENT_USER_QUERY.get() or prompt
        supplemented_chunks = build_supplemental_chunks_context(user_q, system_prompt)
        evidence_with_supplements = (
            f"{system_prompt}\n\n{supplemented_chunks}" if supplemented_chunks else system_prompt
        )
        _CAPTURED_EVIDENCE.set(evidence_with_supplements)

        effective_system_prompt = (
            evidence_with_supplements
            + "\n\n---Critical Analysis Instructions---\n"
            "1. 必须同时精读 `-----Source Documents-----` 与 `-----Supplemental Complete Contract Chunks-----` 中的全部合同条款原文（特别是 part 1/5 至 part 5/5 的完整拼接内容），严格区分目标审查合同（如 HDIL-2026-MC-01 主合同 A / APEX-2026-SUB-MEP-01 分包合同 B）与香港政府通用范本（如 GCC / NTT）。\n"
            "2. 当用户询问某一条款（如“第 4 条”）或是否违反香港法例时，必须结合香港现行法律（如《建造业付款保障条例》Cap. 652、《管制免责条款条例》Cap. 71、《职业安全及健康条例》Cap. 509、《工厂及工业经营条例》Cap. 59、《雇佣条例》Cap. 57 第 32 及 43C 条、《雇员补偿条例》Cap. 282 第 24 条、普通法下惩罚性违约金 Penalty 原则及公共政策 Public Policy）逐款穿透比对。\n"
            "3. 直接给出明确结论，逐条列出【条款编号（如 4.1 / 4.2 / 4.3 / 4.4）+ 原文核心规定 + 违反的具体香港法例或矛盾点】，严禁输出“需具体分析”等套话。"
        )

    messages = []
    if effective_system_prompt:
        messages.append({"role": "system", "content": effective_system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    args_hash = None
    if hashing_kv is not None:
        args_hash = compute_args_hash(active_model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None and cached.get("return"):
            return cached["return"]

    client = AsyncOpenAI(
        api_key="ollama" if use_local_ollama else api_key,
        base_url=active_base_url,
        timeout=3600.0,
        max_retries=2,
        http_client=active_http_client,
    )

    kwargs.setdefault("temperature", 0.1)
    response = await client.chat.completions.create(
        model=active_model,
        messages=messages,
        extra_body={"options": {"num_ctx": 16384}},
        **kwargs,
    )
    content = (response.choices[0].message.content or "").strip()

    if hashing_kv is not None and args_hash is not None and content:
        await hashing_kv.upsert({args_hash: {"return": content, "model": active_model}})
    return content


async def embedding_func(texts: list[str]) -> np.ndarray:
    response = await _EMBED_CLIENT.embeddings.create(
        model=EMBED_MODEL_NAME, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data], dtype=np.float32)


graph_func = HiRAG(
    working_dir=WORK_DIR,
    enable_hierachical_mode=True,
    enable_llm_cache=True,
    chunk_token_size=512,
    embedding_func=EmbeddingFunc(
        embedding_dim=EMBED_DIM, max_token_size=8192, func=embedding_func
    ),
    best_model_func=llm_model_if_cache,
    cheap_model_func=llm_model_if_cache,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_known_entity_ids_and_chunks()
    print(
        f"✅ 已加载 {len(_KNOWN_ENTITY_IDS)} 个图谱节点、{len(_CHUNKS_STORE)} 个文本切片 | "
        f"访问 http://localhost:8000"
    )
    yield
    await _LLM_HTTP_CLIENT.aclose()
    await _EMBED_HTTP_CLIENT.aclose()


app = FastAPI(title="HiRAG Graph Explorer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    api_key: str = ""


def rank_and_filter_nodes(query: str, answer: str, evidence: str) -> list[str]:
    q_upper = query.upper()
    a_upper = answer.upper()
    e_upper = evidence.upper()
    scored = []

    for eid in _KNOWN_ENTITY_IDS:
        if len(eid) < 3 or eid.upper() in _STOP_ENTITIES:
            continue
        eid_u = eid.upper()
        score = 0
        if eid_u in q_upper:
            score += 30
        if eid_u in a_upper:
            score += 20
        elif eid_u in e_upper:
            score += 5
        if score > 0:
            length_bonus = 3 if 4 <= len(eid) <= 45 else 0
            scored.append((score + length_bonus, eid))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_nodes = [eid for s, eid in scored if s >= 15][:12]
    if not top_nodes:
        top_nodes = [eid for _, eid in scored[:8]]
    return top_nodes


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/api/graphml")
async def serve_graphml():
    if not os.path.exists(GRAPHML_PATH):
        raise HTTPException(status_code=404, detail="未找到 graph_chunk_entity_relation.graphml")
    with open(GRAPHML_PATH, "r", encoding="utf-8") as f:
        xml_text = f.read()

    key_match = re.search(r'<key[^>]+attr\.name="source_id"[^>]+id="([^"]+)"', xml_text)
    if not key_match:
        key_match = re.search(r'<key[^>]+id="([^"]+)"[^>]+attr\.name="source_id"', xml_text)
    src_key = key_match.group(1) if key_match else "d2"

    for ref_code in ("HDIL-2026-MC-01", "APEX-2026-SUB-MEP-01"):
        doc_id = _DOC_ALIASES.get(ref_code)
        if doc_id and doc_id in _DOC_TO_CHUNKS:
            all_cids = "&lt;SEP&gt;".join(cid for _, cid, _ in _DOC_TO_CHUNKS[doc_id])
            pattern = (
                rf'(<node id="&quot;{re.escape(ref_code)}&quot;">(?:(?!</node>).)*?<data key="{re.escape(src_key)}">)'
                rf'(.*?)(</data>)'
            )
            xml_text = re.sub(
                pattern,
                lambda m: f'{m.group(1)}&quot;{all_cids}&quot;{m.group(3)}',
                xml_text,
                flags=re.DOTALL,
            )

    return PlainTextResponse(xml_text, media_type="application/xml")


@app.get("/api/chunks")
async def serve_chunks():
    if not _CHUNKS_STORE and os.path.exists(CHUNKS_JSON_PATH):
        load_known_entity_ids_and_chunks()
    return JSONResponse(_CHUNKS_STORE)


@app.post("/api/query")
async def query_graph(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="提问不能为空")

    _CURRENT_API_KEY.set(req.api_key.strip())
    _CURRENT_USER_QUERY.set(query)
    _CAPTURED_EVIDENCE.set("")

    try:
        answer = await graph_func.aquery(query, param=build_safe_query_param())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    evidence = _CAPTURED_EVIDENCE.get()
    matched_nodes = rank_and_filter_nodes(query, answer, evidence)

    return {
        "answer": answer,
        "evidence": evidence,
        "matched_nodes": matched_nodes,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)