"""
규정 Q&A 도우미 (RAG 데모) — Ollama 로컬 버전 (채팅 히스토리 + 예시 버튼)
"""

import re
import numpy as np
import streamlit as st
import pdfplumber
import ollama

DEFAULT_GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"
TOP_K = 4
MAX_CHUNK_LEN = 1200

st.set_page_config(page_title="규정 Q&A 도우미 (로컬)", page_icon="📘", layout="wide")


def extract_pages(pdf_file):
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def chunk_by_article(text, page, max_len=MAX_CHUNK_LEN):
    chunks = []
    pattern = re.compile(r'(제\s*\d+\s*조(?:의\s*\d+)?)')
    matches = list(pattern.finditer(text))
    if not matches:
        for i in range(0, len(text), max_len):
            seg = text[i:i + max_len].strip()
            if seg:
                chunks.append({"article": None, "page": page, "text": seg})
        return chunks
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_no = m.group(1).replace(" ", "")
        body = text[start:end].strip()
        if not body:
            continue
        if len(body) > max_len:
            for i in range(0, len(body), max_len):
                seg = body[i:i + max_len].strip()
                if seg:
                    chunks.append({"article": article_no, "page": page, "text": seg})
        else:
            chunks.append({"article": article_no, "page": page, "text": body})
    return chunks


def build_chunks(pages):
    all_chunks = []
    for page_no, text in pages:
        all_chunks.extend(chunk_by_article(text, page_no))
    return all_chunks


def get_client():
    return ollama.Client(host=OLLAMA_HOST)


def embed_texts(client, texts):
    resp = client.embed(model=EMBED_MODEL, input=texts)
    vecs = np.asarray(resp["embeddings"], dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)


def search(query_vec, matrix, k=TOP_K):
    sims = matrix @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


SYSTEM_PROMPT = """너는 규정 안내 도우미다. 아래 '참고 조항'에 담긴 내용만 근거로 답한다.
규칙:
- 참고 조항에 근거가 없으면 "해당 내용은 규정에서 확인할 수 없습니다"라고 답한다. 추측하지 않는다.
- 답변 끝에 근거가 된 조항 번호를 (근거: 제○조) 형식으로 표기한다.
- 간결하고 명확하게, 실무자가 바로 이해할 수 있게 답한다."""


def generate_answer(client, gen_model, question, retrieved):
    context = "\n\n".join(
        f"[{c['article'] or '본문'} / p.{c['page']}]\n{c['text']}"
        for c in retrieved
    )
    user_msg = f"[참고 조항]\n{context}\n\n[질문]\n{question}"
    resp = client.chat(
        model=gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2, "num_ctx": 8192, "num_predict": 1024},
    )
    return resp["message"]["content"]


st.title("📘 규정 Q&A 도우미 (로컬 / 완전 오프라인)")
st.caption("규정 PDF를 올리면 근거 조항과 함께 답해줍니다. 모든 처리가 로컬에서 실행돼 외부 전송이 없습니다.")

with st.sidebar:
    st.header("설정")
    gen_model = st.text_input(
        "LLM 모델", value=DEFAULT_GEN_MODEL,
        help="미리 'ollama pull'로 받은 모델명. 예: qwen2.5:7b, qwen2.5:3b, exaone3.5:7.8b",
    )
    uploaded = st.file_uploader("규정 PDF 업로드", type=["pdf"])
    st.success("✅ 모든 처리가 로컬에서 실행됩니다.\n외부 전송·API 키 없음.")
    if st.button("🧹 대화 기록 지우기", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if st.button("🔄 인덱스 초기화 (PDF 다시 올리기)", use_container_width=True):
        for k in ("chunks", "matrix", "filename"):
            st.session_state.pop(k, None)
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


def check_ollama(client):
    try:
        client.list()
        return True
    except Exception:
        return False


if uploaded and st.session_state.get("filename") != uploaded.name:
    client = get_client()
    if not check_ollama(client):
        st.error("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요. 설치: https://ollama.com")
        st.stop()
    with st.spinner("PDF 분석 및 인덱싱 중..."):
        pages = extract_pages(uploaded)
        chunks = build_chunks(pages)
        if not chunks:
            st.error("텍스트를 추출하지 못했습니다. (스캔 PDF라면 OCR이 필요합니다)")
            st.stop()
        try:
            matrix = embed_texts(client, [c["text"] for c in chunks])
        except Exception as e:
            st.error(f"임베딩 실패: {e}\n'ollama pull bge-m3'를 먼저 실행했는지 확인하세요.")
            st.stop()
        st.session_state.chunks = chunks
        st.session_state.matrix = matrix
        st.session_state.filename = uploaded.name
    st.success(f"인덱싱 완료 — {len(chunks)}개 조항/청크 ({uploaded.name})")


EXAMPLE_QUESTIONS = [
    "연차휴가는 며칠까지 쓸 수 있나요?",
    "근무시간과 휴게시간은 어떻게 되나요?",
    "출장 여비는 어떻게 신청하나요?",
    "징계의 종류와 절차는 어떻게 되나요?",
]


def answer_question(client, gen_model, question):
    try:
        q_vec = embed_texts(client, [question])[0]
    except Exception as e:
        return None, f"임베딩 실패: {e}"
    hits = search(q_vec, st.session_state.matrix, TOP_K)
    retrieved = [st.session_state.chunks[i] for i, _ in hits]
    try:
        answer = generate_answer(client, gen_model, question, retrieved)
    except Exception as e:
        return None, (f"답변 생성 실패: {e}\n'ollama pull {gen_model}'를 먼저 실행했는지 확인하세요.")
    sources = [
        {"article": c["article"], "page": c["page"], "score": score, "text": c["text"]}
        for (i, score), c in zip(hits, retrieved)
    ]
    return {"answer": answer, "sources": sources}, None


def render_sources(sources):
    with st.expander("🔎 참고한 조항 보기 (출처)"):
        for s in sources:
            st.markdown(f"**{s['article'] or '본문'}** · p.{s['page']} · 유사도 {s['score']:.3f}")
            st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
            st.divider()


if "chunks" in st.session_state:
    st.divider()
    st.markdown("**💡 예시 질문 (클릭해서 바로 질문)**")
    clicked = None
    cols = st.columns(2)
    for idx, ex in enumerate(EXAMPLE_QUESTIONS):
        if cols[idx % 2].button(ex, key=f"ex_{idx}", use_container_width=True):
            clicked = ex
    st.divider()

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_sources(turn["sources"])

    typed = st.chat_input("질문을 입력하세요")
    question = clicked or typed

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner(f"답변 생성 중... ({gen_model})"):
                client = get_client()
                result, err = answer_question(client, gen_model, question)
            if err:
                st.error(err)
            else:
                st.write(result["answer"])
                render_sources(result["sources"])
                st.session_state.history.append({
                    "q": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
else:
    st.info("👈 사이드바에서 규정 PDF를 업로드하면 시작됩니다. (먼저 Ollama 실행 + 모델 pull 필요)")