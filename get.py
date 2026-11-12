"""
纯本地建库与跨合同合规审查脚本：Ollama qwen2.5:7b + bge-m3
核心保障：
1. 绝不重复读取物理文件：直接秒级读取 parsed_texts_cache.json
2. 切换 qwen2.5:7b + num_predict=1536：大幅提升跨合同关系边（Edges）抽取率，消除社区 JSON 截断告警
3. 原生流式防超时 (/api/chat stream=True)：单 token 间隔 90s 超时机制 + 流式实时复读秒级熔断
4. 显存防冲突：Embedding 完成后 15s 自动释放显存，确保 7B 模型高速运行
"""
import os
import re
import sys
import json
import logging
import asyncio
import numpy as np
import pymupdf as fitz
import chardet
from ftfy import fix_text
import httpx
from docx import Document
from striprtf.striprtf import rtf_to_text
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

try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

from hirag import HiRAG, QueryParam
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash, compute_mdhash_id, EmbeddingFunc

logger = logging.getLogger("HiRAG")

# ==========================================
# 0. 纯本地配置（7B 高质量抽取 + 原生流式防超时）
# ==========================================
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_BASE = f"{OLLAMA_HOST}/v1"

LLM_MODEL   = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = "bge-m3:latest"
EMBED_DIM   = 1024

LLM_STREAM_READ_TIMEOUT = 90.0
EMBED_TIMEOUT           = 120.0

LLM_NUM_CTX           = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 4096))
LLM_NUM_PREDICT       = 1536
LLM_TEMPERATURE       = 0.1
LLM_REPEAT_PENALTY    = 1.15
LLM_FREQUENCY_PENALTY = 0.1
LLM_MAX_RETRIES       = 3

WORK_DIR          = "./pdf_work_dir_local"
PARSED_CACHE_FILE = os.path.join(WORK_DIR, "parsed_texts_cache.json")

_SHARED_HTTP_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=httpx.Timeout(connect=30.0, read=LLM_STREAM_READ_TIMEOUT, write=30.0, pool=30.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)

SKIP_PREFIXES = ("~$", ".")


# ==========================================
# 1. LLM 函数（原生流式 + 实时复读熔断 + 7B 独立缓存）
# ==========================================
def _has_severe_repetition(text: str) -> bool:
    if not text or len(text) < 200:
        return False
    return bool(re.search(r"(.{20,}?)\1{4,}", text, flags=re.DOTALL))


async def llm_model_if_cache(prompt, system_prompt="", history_messages=[], **kwargs):
    hashing_kv = kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    args_hash = None
    if hashing_kv is not None:
        args_hash = compute_args_hash(LLM_MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None and not _has_severe_repetition(cached.get("return", "")):
            return cached["return"]

    last_err = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            options = {
                "num_ctx": LLM_NUM_CTX,
                "num_predict": LLM_NUM_PREDICT,
                "temperature": min(0.5, LLM_TEMPERATURE + 0.12 * attempt),
                "repeat_penalty": min(1.35, LLM_REPEAT_PENALTY + 0.08 * attempt),
                "frequency_penalty": LLM_FREQUENCY_PENALTY,
            }
            payload = {
                "model": LLM_MODEL,
                "messages": messages,
                "stream": True,
                "keep_alive": "10m",
                "options": options,
            }

            chunks_buf = []
            char_count_since_check = 0
            async with _SHARED_HTTP_CLIENT.stream(
                "POST", f"{OLLAMA_HOST}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        chunks_buf.append(piece)
                        char_count_since_check += len(piece)
                        if char_count_since_check >= 60:
                            char_count_since_check = 0
                            if _has_severe_repetition("".join(chunks_buf)):
                                raise RuntimeError("Detected severe repetition loop in stream")
                    if data.get("done"):
                        break

            content = "".join(chunks_buf).strip()
            if _has_severe_repetition(content):
                raise RuntimeError("Detected severe repetition loop in LLM output")

            if hashing_kv is not None and args_hash is not None and content:
                await hashing_kv.upsert(
                    {args_hash: {"return": content, "model": LLM_MODEL}}
                )
            return content
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_retryable = any(
                k in err_str.lower()
                for k in (
                    "500",
                    "502",
                    "503",
                    "429",
                    "timeout",
                    "timed out",
                    "connect",
                    "closed",
                    "repeat",
                    "repetition",
                )
            )
            if is_retryable and attempt < LLM_MAX_RETRIES:
                logger.warning(
                    f"⚠️ 第 {attempt + 1} 次调用异常（{err_str[:120]}），3s 后调参重试..."
                )
                await asyncio.sleep(3)
                continue
            raise
    raise last_err


# ==========================================
# 2. Embedding（本地 bge-m3，15s 自动释放显存）
# ==========================================
async def embedding_func(texts: list[str]) -> np.ndarray:
    resp = await _SHARED_HTTP_CLIENT.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts, "keep_alive": "15s"},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"], dtype=np.float32)


# ==========================================
# 3. 初始化 HiRAG
# ==========================================
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
    embedding_batch_num=2,
    embedding_func_max_async=1,
    best_model_max_async=1,
    cheap_model_max_async=1,
)


# ==========================================
# 4. 文档清洗与单次读取缓存机制
# ==========================================
def clean_text(text: str, filename: str = "") -> str:
    if not text:
        return ""
    text = fix_text(text)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\d{2,})([A-Z][a-z])", r"\1\n\2", text)
    text = re.sub(r"([。！？!?])\s*([^\n])", r"\1\n\2", text)

    text = re.sub(r"(\b\d{1,4}[\.\s]+){20,}", "", text)
    text = re.sub(r"(\b1\s+){10,}", "", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if filename:
        doc_tag = os.path.splitext(filename)[0]
        text = re.sub(
            r"(?m)^(\s*(?:(?:S?\d+\.)+\d*\s+[A-Z]|SCHEDULE\s+\d+|SECTION\s+\d+|CLAUSE\s+\d+|PART\s+[A-Z0-9]+))",
            rf"[{doc_tag}] \1",
            text,
        )
        text = f"[DOCUMENT_SOURCE: {filename}]\n{text.strip()}"

    return text.strip()


def parse_pdf(filepath):
    doc = fitz.open(filepath)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def parse_docx(filepath):
    doc = Document(filepath)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            dedup_cells = []
            for cell in row.cells:
                c_txt = cell.text.strip().replace("\n", " ")
                if c_txt and (not dedup_cells or dedup_cells[-1] != c_txt):
                    dedup_cells.append(c_txt)
            if dedup_cells:
                parts.append(" | ".join(dedup_cells))
    return "\n".join(parts)


def parse_txt(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)["encoding"]
    return raw.decode(enc or "utf-8", errors="ignore")
def parse_rtf(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        rtf_content = f.read()
    return rtf_to_text(rtf_content)

def parse_doc(filepath, word_app=None):
    if not HAS_WIN32:
        raise NotImplementedError("缺少 pywin32，无法解析 .doc")
    own_app = word_app is None
    if own_app:
        word_app = win32com.client.Dispatch("Word.Application")
        word_app.Visible = False
    doc = None
    try:
        doc = word_app.Documents.Open(os.path.abspath(filepath), ReadOnly=True)
        return doc.Content.Text
    finally:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if own_app:
            word_app.Quit()


def load_or_parse_docs_once(docs_dir: str, cache_file: str) -> list[str]:
    cached_map = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    cached_map[item["filename"]] = item
        except Exception as e:
            print(f"⚠️ 缓存文件读取异常 ({e})，将重新建立缓存...")

    records = []
    cache_updated = False
    word_app = None

    try:
        for filename in sorted(os.listdir(docs_dir)):
            if any(filename.lower().startswith(p.lower()) for p in SKIP_PREFIXES):
                continue
            filepath = os.path.join(docs_dir, filename)
            if os.path.isdir(filepath):
                continue
            ext = filename.lower().split(".")[-1]
            if ext not in ("pdf", "docx", "txt", "doc", "rtf"):
                continue

            stat = os.stat(filepath)
            sig = f"{stat.st_size}_{int(stat.st_mtime)}"

            if filename in cached_map and cached_map[filename].get("sig") == sig:
                text = cached_map[filename]["text"]
                records.append(cached_map[filename])
                print(f"⚡ [缓存直读] 跳过物理文件读取: {filename} ({len(text)} 字符)")
                continue

            try:
                if ext == "pdf":
                    raw = parse_pdf(filepath)
                elif ext == "docx":
                    raw = parse_docx(filepath)
                elif ext == "txt":
                    raw = parse_txt(filepath)
                elif ext == "doc":
                    if word_app is None:
                        if not HAS_WIN32:
                            raise NotImplementedError("缺少 pywin32，无法解析 .doc")
                        word_app = win32com.client.Dispatch("Word.Application")
                        word_app.Visible = False
                    raw = parse_doc(filepath, word_app=word_app)
                elif ext == "rtf":
                    raw = parse_rtf(filepath)
                cleaned = clean_text(raw, filename)
                if cleaned and len(cleaned) > 10:
                    entry = {"filename": filename, "sig": sig, "text": cleaned}
                    records.append(entry)
                    cache_updated = True
                    print(f"✅ [首次解析] 成功提取并清洗: {filename} ({len(cleaned)} 字符)")
                else:
                    print(f"⚠️ 清洗后内容过短，忽略: {filename}")
            except Exception as e:
                print(f"❌ 解析 {filename} 失败: {e}")
    finally:
        if word_app is not None:
            try:
                word_app.Quit()
            except Exception:
                pass

    if cache_updated and records:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"💾 解析缓存已固化至: {cache_file}")

    return [r["text"] for r in records]


# ==========================================
# 5. 终端进度管理器与安全落盘
# ==========================================
async def flush_all_storages(gf: HiRAG):
    storage_names = [
        "full_docs",
        "text_chunks",
        "llm_response_cache",
        "community_reports",
        "chunks_vdb",
        "entities_vdb",
        "relationships_vdb",
        "communities_vdb",
        "chunk_entity_relation_graph",
    ]
    for name in storage_names:
        storage = getattr(gf, name, None)
        if storage is not None and hasattr(storage, "index_done_callback"):
            await storage.index_done_callback()


class TerminalManager:
    def __init__(self, original_stream):
        self.original = original_stream
        self.progress_text = "⏳ 准备开始..."
        self.save_event = asyncio.Event()
        self.last_pct = -1
        self.saved = False

    def write(self, text):
        text = text.replace("\r", "")
        if "Processed" in text and "chunks" in text:
            match = re.search(
                r"Processed\s+(\d+)\((\d+)%\)\s+chunks,\s+(\d+)\s+entities.*?,\s+(\d+)\s+relations",
                text,
            )
            if match:
                pct = int(match.group(2))
                self.progress_text = (
                    f"⏳ 进度: {match.group(1)} chunks ({pct}%), "
                    f"{match.group(3)} 实体, {match.group(4)} 关系"
                )
                if pct != self.last_pct:
                    self.last_pct = pct
                    self.saved = False
                    self.save_event.set()
                self._render()
                return
        if "落盘成功" in text:
            self.saved = True
            self._render()
            return
        if text.strip():
            self._clear_line()
            self.original.write(text)
            if not text.endswith("\n"):
                self.original.write("\n")
            self.original.flush()
            self._render()

    def _render(self):
        self._clear_line()
        msg = self.progress_text + (" | 💾 缓存已保存" if self.saved else " | ⏳ 正在落盘...")
        self.original.write(f"\r{msg.ljust(100)}")
        self.original.flush()

    def _clear_line(self):
        self.original.write("\r" + " " * 100 + "\r")
        self.original.flush()

    def flush(self):
        self.original.flush()

    def isatty(self):
        return True


# ==========================================
# 6. 主流程
# ==========================================
async def main():
    print("🔧 纯本地 LLM 路由（7B 流式高连通度建库模式）：")
    print(
        f"   LLM: {LLM_MODEL} (num_ctx={LLM_NUM_CTX}, num_predict={LLM_NUM_PREDICT}) @ {OLLAMA_HOST}"
    )
    print(f"   Embedding: {EMBED_MODEL} (keep_alive=15s 自动释放显存) @ {OLLAMA_HOST}")
    print(
        f"   防复读: temperature={LLM_TEMPERATURE}, "
        f"repeat_penalty={LLM_REPEAT_PENALTY}, "
        f"frequency_penalty={LLM_FREQUENCY_PENALTY}"
    )
    print()

    docs_dir = r"D:\HiRAG\test"
    if not os.path.exists(docs_dir):
        print(f"❌ 目录不存在: {docs_dir}")
        return

    texts = load_or_parse_docs_once(docs_dir, PARSED_CACHE_FILE)
    if not texts:
        print("❌ 没有可用文档，退出。")
        return

    pending_texts = []
    for t in texts:
        doc_id = compute_mdhash_id(t.strip(), prefix="doc-")
        existing = await graph_func.full_docs.get_by_id(doc_id)
        if existing is None:
            pending_texts.append(t)

    if not pending_texts:
        print(f"\n✅ 全部 {len(texts)} 份文档已在本地库中构建完毕，无需重复建库！\n")
        await _SHARED_HTTP_CLIENT.aclose()
        return

    print(
        f"\n📄 共加载 {len(texts)} 份文档（其中 {len(pending_texts)} 份待入库/断点续传），开始本地建库...\n"
    )

    manager = TerminalManager(sys.stdout)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = manager
    sys.stderr = manager

    async def monitor_and_save():
        while True:
            await manager.save_event.wait()
            manager.save_event.clear()
            try:
                if hasattr(graph_func.llm_response_cache, "index_done_callback"):
                    await graph_func.llm_response_cache.index_done_callback()
                manager.write("落盘成功\n")
            except Exception:
                pass

    monitor_task = asyncio.create_task(monitor_and_save())

    try:
        await graph_func.ainsert(pending_texts)
        await flush_all_storages(graph_func)
        manager._clear_line()
        print(f"\n🎉 建库完成！文件已保存在 {WORK_DIR}\n")
    except Exception as e:
        manager._clear_line()
        print(f"\n❌ 建库中途中断: {e}")
        print("💡 已完成切片的推理结果已实时落盘至 llm_response_cache，下次运行直接秒级续传。\n")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        monitor_task.cancel()
        try:
            await flush_all_storages(graph_func)
        except Exception:
            pass
        await _SHARED_HTTP_CLIENT.aclose()

    print("\n--- 建库阶段结束 ---")


if __name__ == "__main__":
    asyncio.run(main())