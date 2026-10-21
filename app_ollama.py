"""
규정 도우미 — 규정 PDF Q&A (RAG, Ollama 로컬 버전)
─────────────────────────────────────────────
LLM과 임베딩을 모두 '개인 PC 안에서' 실행. 데이터가 외부로 나가지 않음.

[ 사전 준비 ]
1) Ollama 설치 (https://ollama.com)
2) 모델 받기 (cmd)
   ollama pull bge-m3
   ollama pull qwen2.5:7b
3) python -m pip install -r requirements_ollama.txt

[ 기능 ]
- 같은 폴더의 '취업규칙.pdf' 자동 로드 + 분석 결과 캐시(.cache)
- 앱 시작 시 모델 예열, 답변 스트리밍, 응답 시간·근거 일치도 표시
- 이어서 질문하기: 앞 대화를 기억해 후속 질문에 답함
- 근거를 못 찾으면 지어내지 않고 안내
- 사이드바: 상태 표시, 모델 선택, 조항 직접 보기, 대화 내용 저장

[ 실행 ]
python -m streamlit run app_ollama.py
"""

import hashlib
import io
import itertools
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
import pdfplumber
import streamlit as st

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DEFAULT_GEN_MODEL = "qwen2.5:7b"     # 답변 생성 모델(기본 선택)
EMBED_MODEL = "bge-m3"               # 임베딩 모델
OLLAMA_HOST = "http://localhost:11434"
TOP_K = 4                            # 질문당 참고할 조항 수
MAX_CHUNK_LEN = 1200
KEEP_ALIVE = "2h"                    # 모델을 메모리에 유지하는 시간
HISTORY_TURNS = 2                    # 이어서 질문할 때 기억할 직전 대화 수
DEFAULT_MIN_SCORE = 0.35             # 이 값보다 관련도가 낮으면 '근거 없음' 처리 (사이드바에서 조절)
REL_GAP = 0.10                       # 1등 조항보다 이만큼 이상 관련도가 낮은 조항은 버림
KEYWORD_WEIGHT = 0.3                 # 검색 순위에서 단어 일치 점수 비중 (나머지는 의미 점수)

APP_DIR = Path(__file__).parent
DEFAULT_PDF = APP_DIR / "취업규칙.pdf"
CACHE_DIR = APP_DIR / ".cache"

# 기본 문서(취업규칙.pdf)용 예시 질문 — 시연용으로 미리 검증한 질문
EXAMPLE_QUESTIONS = [
    "연차휴가는 며칠까지 쓸 수 있나요?",
    "근무시간과 휴게시간은 어떻게 되나요?",
    "수습기간은 얼마인가요?",
    "징계의 종류와 절차는 어떻게 되나요?",
]
# 다른 문서를 올렸는데 예시 질문 자동 생성이 실패했을 때 쓰는 질문
FALLBACK_EXAMPLES = [
    "이 규정의 핵심 내용은 무엇인가요?",
    "가장 중요한 조항은 무엇인가요?",
]
NO_GROUNDS_MSG = ("해당 내용은 규정에서 확인할 수 없습니다. "
                  "질문과 관련된 조항을 찾지 못했어요. 규정에 있는 내용으로 다시 질문해 주세요.")

st.set_page_config(page_title="규정 도우미", page_icon="🔒", layout="wide")


# ─────────────────────────────────────────────
# 1) PDF 파싱 + 조항 단위 청킹
# ─────────────────────────────────────────────
def extract_pages(pdf_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


# 조항 시작: 줄 맨 앞의 '제N조(' 또는 '제N조의M(' 만 인정.
# → 본문 속 "근로기준법 제60조에 따라" 같은 인용은 조항 시작으로 보지 않음.
ARTICLE_RE = re.compile(r'(?m)^[ \t]*(제\s*\d+\s*조(?:의\s*\d+)?)\s*\(')
TITLE_RE = re.compile(r'^\s*(제\s*\d+\s*조(?:의\s*\d+)?)\s*\(([^)]*)\)')
HEADING_RE = re.compile(r'^\s*(제\s*\d+\s*[장절]|부\s*칙)')
CHUNK_VERSION = "3"   # 청킹 방식이 바뀌면 올려서 예전 캐시를 무효화


def _split(text, max_len=MAX_CHUNK_LEN):
    return [text[i:i + max_len].strip() for i in range(0, len(text), max_len)
            if text[i:i + max_len].strip()]


def _ordered(matches):
    """조항 번호는 앞으로만 진행한다고 보고, 거꾸로 가는 번호는 본문 속 인용으로 간주해 버림.
    예) 제62조 다음 줄 맨 앞에 '제14조(…)'가 오면, 줄바꿈된 법령 인용이므로 무시.
    단, 제1조로 다시 시작하는 경우(부칙 등)는 새 조항으로 인정."""
    kept, prev = [], (0, 0)
    for m in matches:
        nums = re.findall(r"\d+", m.group(1))
        cur = (int(nums[0]), int(nums[1]) if len(nums) > 1 else 0)
        if cur >= prev or cur == (1, 0):
            kept.append(m)
            prev = cur
    return kept


def build_chunks(pages):
    """문서 전체를 이어 붙인 뒤 조항 단위로 자름. (페이지를 넘어가는 조항도 온전히 유지)"""
    if not pages:
        return []
    full, offsets = "", []
    for page_no, text in pages:
        offsets.append((len(full), page_no))
        full += text + "\n"

    def page_at(pos):
        page = offsets[0][1]
        for off, no in offsets:
            if off > pos:
                break
            page = no
        return page

    matches = _ordered(ARTICLE_RE.finditer(full))
    if not matches:  # 조항 구조가 없는 문서: 페이지별 길이 분할
        return [{"article": None, "page": no, "text": seg}
                for no, text in pages for seg in _split(text)]

    chunks = []
    for seg in _split(full[:matches[0].start()]):
        chunks.append({"article": None, "page": offsets[0][1], "text": seg})
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        article_no = m.group(1).replace(" ", "")
        for seg in _split(full[m.start():end]):
            chunks.append({"article": article_no, "page": page_at(m.start()), "text": seg})
    return chunks


def article_list(chunks):
    """조항 보기용 목록: [(표시이름, 페이지, 본문, 포함된 조각 번호들)].
    연달아 붙은 같은 조항의 조각은 합치고, 같은 번호가 다시 나오면(예: 부칙 제1조) 따로 둠."""
    items, prev = [], None
    for ci, c in enumerate(chunks):
        art = c["article"]
        if not art:
            prev = None
            continue
        m = TITLE_RE.match(c["text"])
        if art == prev and not m:      # 긴 조항이 잘린 뒷부분 → 앞 조각에 이어 붙임
            items[-1]["text"] += "\n" + c["text"]
            items[-1]["chunks"].add(ci)
        else:
            label = f"{m.group(1).replace(' ', '')}({m.group(2).strip()})" if m else art
            items.append({"label": label, "page": c["page"], "text": c["text"], "chunks": {ci}})
        prev = art
    out = []
    for it in items:
        body = "\n".join(l for l in it["text"].split("\n") if not HEADING_RE.match(l))
        out.append((it["label"], it["page"], body.strip(), it["chunks"]))
    return out


def article_index_of(chunk_id):
    """조각 번호 → 조항 목록에서의 위치 (없으면 None)."""
    for i, a in enumerate(st.session_state.get("articles", [])):
        if chunk_id in a[3]:
            return i
    return None


def goto_article(idx):
    """'전체 보기' 버튼: 사이드바 조항 보기를 해당 조항으로 이동시키고 펼침."""
    st.session_state.article_pick = idx
    st.session_state.article_open = True
    st.session_state.goto_toast = st.session_state.articles[idx][0]


# ─────────────────────────────────────────────
# 2) Ollama 연결 · 상태 확인 · 예열
# ─────────────────────────────────────────────
def get_client():
    return ollama.Client(host=OLLAMA_HOST)


def _full_name(name):
    """'bge-m3' → 'bge-m3:latest' 처럼 태그를 붙여 비교용 이름으로 맞춤."""
    return name if ":" in name else f"{name}:latest"


def list_models(client):
    """(연결 여부, 설치된 모델 이름 목록)"""
    try:
        resp = client.list()
    except Exception:
        return False, []
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names = []
    for m in models or []:
        n = getattr(m, "model", None)
        if n is None and isinstance(m, dict):
            n = m.get("model") or m.get("name")
        if n:
            names.append(n)
    return True, names


def chat(client, **kwargs):
    """생각 모드(thinking)를 끄고 호출. 생각 모드를 지원하지 않는 모델(qwen2.5 등)은
    'think' 옵션 없이 다시 호출한다. → qwen3.5 같은 최신 모델도 답이 바로 나오게 함.
    스트리밍은 첫 조각을 받을 때 오류가 나므로, 첫 조각을 미리 받아 확인한다."""
    def call(**extra):
        resp = client.chat(**kwargs, **extra)
        if kwargs.get("stream"):
            it = iter(resp)
            try:
                first = next(it)
            except StopIteration:
                return iter(())
            return itertools.chain([first], it)
        return resp

    try:
        return call(think=False)
    except TypeError:          # think 옵션을 모르는 구버전 ollama 라이브러리
        return call()
    except Exception as e:
        if "think" in str(e).lower():
            return call()
        raise


@st.cache_resource(show_spinner=False)
def warm_up(gen_model):
    """모델을 미리 메모리에 올려 첫 질문도 빠르게 답하도록 함. (모델별 1회)"""
    client = get_client()
    client.embed(model=EMBED_MODEL, input=["예열"], keep_alive=KEEP_ALIVE)
    chat(client,
         model=gen_model,
         messages=[{"role": "user", "content": "안녕"}],
         options={"num_predict": 1},
         keep_alive=KEEP_ALIVE)
    return True


# ─────────────────────────────────────────────
# 3) 임베딩 · 인덱싱 (디스크 캐시)
# ─────────────────────────────────────────────
def embed_texts(client, texts):
    resp = client.embed(model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE)
    vecs = np.asarray(resp["embeddings"], dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)


def build_index(client, pdf_bytes):
    """PDF → (청크 목록, 임베딩 행렬). 같은 PDF는 .cache에서 바로 불러옴."""
    key = hashlib.sha256(pdf_bytes + EMBED_MODEL.encode() + CHUNK_VERSION.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data["chunks"], data["matrix"]
        except Exception:
            pass  # 캐시가 깨졌으면 새로 만듦

    chunks = build_chunks(extract_pages(pdf_bytes))
    if not chunks:
        return [], None
    matrix = embed_texts(client, [c["text"] for c in chunks])

    try:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"chunks": chunks, "matrix": matrix}, f)
    except Exception:
        pass
    return chunks, matrix


def load_document(client, pdf_bytes, name, is_default=False):
    """문서를 인덱싱해 세션에 올림. 성공하면 True. (is_default: 자동 로드한 기본 문서 여부)"""
    t0 = time.perf_counter()
    try:
        chunks, matrix = build_index(client, pdf_bytes)
    except Exception as e:
        st.error(f"문서 분석 실패: {e}\n'ollama pull {EMBED_MODEL}'를 실행했는지 확인하세요.")
        return False
    if not chunks:
        st.error("텍스트를 추출하지 못했습니다. (스캔 PDF라면 OCR이 필요합니다)")
        return False
    st.session_state.chunks = chunks
    st.session_state.matrix = matrix
    st.session_state.kw_index = KeywordIndex([c["text"] for c in chunks])
    st.session_state.articles = article_list(chunks)
    st.session_state.doc_name = name
    st.session_state.doc_is_default = is_default
    st.session_state.doc_sig = hashlib.sha256(pdf_bytes).hexdigest()
    st.session_state.history = []
    st.session_state.index_sec = time.perf_counter() - t0
    return True


# ─────────────────────────────────────────────
# 4) 검색 · 답변 생성
# ─────────────────────────────────────────────
FOLLOWUP_CUES = ("그럼", "그러면", "그런데", "근데", "그리고", "그래서", "그건", "그거", "그것",
                 "그 경우", "이 경우", "그때", "그 때", "이건", "이거", "만약", "혹시", "또 ", "그렇다면")


def is_followup(question, history):
    """앞 대화에 이어지는 질문인지 판단: 이어주는 말로 시작하거나 아주 짧은 질문."""
    if not history:
        return False
    q = question.strip()
    return q.startswith(FOLLOWUP_CUES) or len(q.replace(" ", "")) <= 8


# ── 표현 사전: 직원이 쓰는 말 → 규정에 적힌 말 (검색할 때 같이 찾음) ──
# 필요하면 여기에 줄을 추가하면 됨. 예) "재택": "재택근무 근로장소",
SYNONYMS = {
    "근무시간": "근로시간", "근무 시간": "근로시간", "업무시간": "근로시간", "출퇴근": "시업 종업 시각",
    "점심": "휴게시간", "쉬는 시간": "휴게시간", "쉬는시간": "휴게시간",
    "월급": "임금 지급", "급여": "임금", "봉급": "임금", "월급날": "임금 지급일", "급여일": "임금 지급일",
    "보너스": "상여금", "성과급": "상여금",
    "야근": "연장근로 야간근로", "잔업": "연장근로", "초과근무": "연장근로", "추가근무": "연장근로",
    "주말": "휴일", "주말근무": "휴일근로", "특근": "휴일근로",
    "여름휴가": "하기휴가", "여름 휴가": "하기휴가", "휴가비": "하기휴가 상여금",
    "연차": "연차유급휴가", "월차": "연차유급휴가",
    "결혼": "경조사 휴가 본인 결혼", "장례": "경조사 휴가 사망", "조사": "경조사", "경사": "경조사",
    "출산휴가": "출산전후휴가", "아내 출산": "배우자 출산휴가", "와이프": "배우자",
    "아파서": "병가 질병", "아프면": "병가 질병", "몸이 안 좋": "병가 질병",
    "그만두": "퇴직", "그만 두": "퇴직", "사직": "퇴직", "사표": "퇴직", "퇴사": "퇴직",
    "잘리": "해고", "짤리": "해고", "해임": "해고",
    "신입": "신규 채용 수습", "인턴": "수습",
    "벌점": "징계", "처벌": "징계", "불복": "재심",
    "갑질": "직장 내 괴롭힘", "괴롭힘": "직장 내 괴롭힘",
    "성추행": "직장 내 성희롱", "성희롱": "직장 내 성희롱",
    "건강검진": "건강진단", "검진": "건강진단",
    "육아": "육아휴직 육아기", "임신": "임산부 임신 중",
    "늦게 출근": "지각", "일찍 퇴근": "조퇴",
    "부서 이동": "전직 배치", "승진": "승진", "인사이동": "전직 배치",
    "정년퇴직": "정년", "은퇴": "정년",
}


def expand_query(q):
    """질문에 직원 표현이 있으면 규정 용어를 덧붙임. 예) '월급날은?' → '월급날은? 임금 지급일'"""
    extra = [v for k, v in SYNONYMS.items() if k in q]
    return f"{q} {' '.join(dict.fromkeys(extra))}" if extra else q


# ── 단어 일치 검색 (BM25, 한국어는 글자 2개 단위로 쪼개 비교) ──
def _tokens(text):
    """'연차유급휴가는' → ['연차', '차유', '유급', '급휴', '휴가', '가는'] (띄어쓰기 단위 안에서 2글자씩)"""
    toks = []
    for w in re.findall(r"[가-힣A-Za-z0-9]+", text):
        if len(w) == 1:
            continue
        toks += [w[i:i + 2] for i in range(len(w) - 1)]
    return toks


class KeywordIndex:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.docs = [_tokens(t) for t in texts]
        self.lens = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = self.lens.mean() if len(self.lens) else 1.0
        self.k1, self.b = k1, b
        self.tf = []
        df = {}
        for d in self.docs:
            counts = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: np.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for t in set(_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


def search(client, question, history, matrix, kw_index=None, k=TOP_K):
    """의미 검색(임베딩) + 단어 일치 검색(BM25)을 섞어 조항 순위를 매김.
    - 이어지는 질문이면 직전 질문과 합친 문장으로도 찾아 더 관련 높은 쪽을 채택
      (이어지는 질문이 아니면 합치지 않음: 엉뚱한 질문이 앞 대화 덕에 점수가 오르는 것 방지)
    반환: (조각 번호, 검색 점수) 목록, 의미 점수 최고값(=근거 일치도)"""
    base = [question]
    if is_followup(question, history):
        base.append(f"{history[-1]['q']} {question}")
    expanded = [expand_query(q) for q in base]          # 직원 표현 → 규정 용어 보강
    texts = list(dict.fromkeys(base + expanded))
    vecs = dict(zip(texts, embed_texts(client, texts)))
    sim = lambda qs: (matrix @ np.stack([vecs[q] for q in qs]).T).max(axis=1)
    # 근거 일치도는 '사용자가 실제로 쓴 질문'으로만 판단
    # (표현 사전 때문에 '점심 메뉴 추천' 같은 엉뚱한 질문이 통과되는 것 방지)
    top_dense = float(sim(base).max()) if len(matrix) else 0.0
    dense = np.maximum(sim(base), sim(expanded))        # 순위는 보강한 질문까지 포함
    queries = expanded

    score = dense
    if kw_index is not None and KEYWORD_WEIGHT > 0:
        kw = np.max([kw_index.scores(q) for q in queries], axis=0)
        if kw.max() > 0:
            score = (1 - KEYWORD_WEIGHT) * dense + KEYWORD_WEIGHT * (kw / kw.max()) * max(top_dense, 0)

    idx = np.argsort(-score)[:k]
    hits = [(int(i), float(score[i])) for i in idx]
    # 1등보다 점수가 크게 떨어지는 조항은 제외 (엉뚱한 출처 노출·불필요한 읽기 시간 방지)
    if hits:
        hits = [h for h in hits if h[1] >= hits[0][1] - REL_GAP]
    return hits, top_dense


ARTICLE_REF_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")


def cited_articles(answer):
    """답변에 적힌 근거 조항 번호 집합. 예) '(근거: 제33조, 제34조의2)' → {'제33조', '제34조의2'}"""
    return {f"제{a}조" + (f"의{b}" if b else "") for a, b in ARTICLE_REF_RE.findall(answer or "")}


def article_label(chunk):
    m = TITLE_RE.match(chunk["text"])
    if m:
        return f"{m.group(1).replace(' ', '')}({m.group(2).strip()})"
    return chunk["article"] or "본문"


SYSTEM_PROMPT = """너는 회사 규정 안내 도우미다. 반드시 한국어로 답한다.
'참고 조항'에 적힌 내용만 근거로 답하고, 그 밖의 법령·상식·추측은 덧붙이지 않는다.

답변 방법:
1. 첫 문장에 질문에 대한 결론을 바로 말한다.
2. 숫자·기간·금액·조건(예: 15일, 3개월, 80퍼센트 이상)은 조항에 적힌 그대로 옮긴다. 바꾸거나 계산해서 새 숫자를 만들지 않는다.
3. 예외나 조건이 있으면 빠뜨리지 말고 함께 적는다. (예: "다만, ~한 경우에는 ~")
4. 여러 조항이 관련되면 각각 나눠서 설명한다.
5. 참고 조항에 답이 없으면 "해당 내용은 규정에서 확인할 수 없습니다"라고만 답한다.
6. 앞선 대화가 있으면 그 흐름을 이어서 답한다. (예: "그럼 3년 다니면?"은 직전 질문 주제의 후속 질문)
7. 마지막 줄에 실제로 근거로 쓴 조항만 (근거: 제○조, 제○조) 형식으로 적는다."""


def build_messages(question, retrieved, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-HISTORY_TURNS:]:          # 직전 대화(질문·답변)만 전달
        msgs.append({"role": "user", "content": turn["q"]})
        msgs.append({"role": "assistant", "content": turn["answer"]})
    context = "\n\n".join(
        f"[{c['article'] or '본문'} / p.{c['page']}]\n{c['text']}" for c in retrieved
    )
    msgs.append({"role": "user", "content": f"[참고 조항]\n{context}\n\n[질문]\n{question}"})
    return msgs


def stream_answer(client, gen_model, messages):
    """답변을 글자 단위로 흘려보내는 제너레이터."""
    stream = chat(client,
                  model=gen_model,
                  messages=messages,
                  options={"temperature": 0.1, "num_ctx": 8192, "num_predict": 1024},
                  keep_alive=KEEP_ALIVE,
                  stream=True)
    for chunk in stream:
        piece = chunk["message"]["content"]   # 생각 과정(thinking)은 별도 칸이라 화면에 안 나옴
        if piece:
            yield piece


# ─────────────────────────────────────────────
# 5) 예시 질문 자동 생성 (업로드한 문서에 맞춤)
# ─────────────────────────────────────────────
EXAMPLE_GEN_PROMPT = """너는 규정 문서를 보고 직원이 실제로 궁금해할 질문을 만드는 역할이다.
아래는 한 규정 문서에서 뽑은 조항 제목 목록이다.
이 목록에 담긴 내용만 근거로, 직원이 실제로 물어볼 법한 자연스러운 질문 4개를 만들어라.
규칙:
- 반드시 목록에 있는 내용에 근거한 질문만 만든다. 목록에 없는 주제는 만들지 않는다.
- 서로 다른 주제로 4개를 만든다.
- 한 줄에 질문 하나씩, 번호·기호·따옴표 없이 질문 문장만 출력한다.
- 간결한 한국어 존댓말로, 물음표로 끝낸다."""


def collect_article_titles(articles, limit=30):
    """조항 제목을 문서 전체에서 고르게 뽑음 (앞부분에만 치우치지 않도록)."""
    labels = list(dict.fromkeys(a[0] for a in articles))
    if len(labels) <= limit:
        return labels
    picks = np.linspace(0, len(labels) - 1, limit).round().astype(int)   # 처음·끝 포함, 같은 간격
    return [labels[i] for i in dict.fromkeys(picks)]


def clean_question(line):
    """AI 출력 한 줄에서 번호·기호·따옴표를 제거."""
    line = re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', '', line.strip())
    return line.strip().strip('"\'“”')


def generate_example_questions(client, gen_model, articles):
    """문서의 조항 제목을 보고 예시 질문 4개를 생성. 실패하면 기본 질문."""
    titles = collect_article_titles(articles)
    if not titles:
        return FALLBACK_EXAMPLES
    try:
        resp = chat(client,
                    model=gen_model,
                    messages=[{"role": "system", "content": EXAMPLE_GEN_PROMPT},
                              {"role": "user", "content": "[조항 제목 목록]\n" + "\n".join(f"- {t}" for t in titles)}],
                    options={"temperature": 0.4, "num_ctx": 4096, "num_predict": 256},
                    keep_alive=KEEP_ALIVE)
        content = resp["message"]["content"]
    except Exception:
        return FALLBACK_EXAMPLES
    qs = [clean_question(l) for l in content.splitlines()]
    qs = list(dict.fromkeys(q for q in qs if q.endswith("?") and len(q) >= 6))
    return qs[:4] if qs else FALLBACK_EXAMPLES


def get_examples(client, gen_model):
    """현재 문서의 예시 질문. 기본 문서는 검증된 질문, 그 외는 자동 생성(디스크에 저장해 재사용)."""
    if st.session_state.get("doc_is_default"):
        return EXAMPLE_QUESTIONS
    key = f"examples_{st.session_state.doc_sig[:16]}_{re.sub(r'[^A-Za-z0-9]', '_', gen_model)}"
    if key in st.session_state:
        return st.session_state[key]
    cache_file = CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        qs = [l for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        with st.spinner("이 문서에 맞는 예시 질문을 만드는 중... (문서마다 처음 한 번만)"):
            qs = generate_example_questions(client, gen_model, st.session_state.articles)
        if qs is not FALLBACK_EXAMPLES:
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                cache_file.write_text("\n".join(qs), encoding="utf-8")
            except Exception:
                pass
    st.session_state[key] = qs
    return qs


# ─────────────────────────────────────────────
# 6) 화면 출력 도우미
# ─────────────────────────────────────────────
def _source_block(s, key):
    head, btn = st.columns([4, 1])
    head.markdown(f"**{s['label']}** · p.{s['page']} · 관련도 {s['score']:.2f}")
    idx = article_index_of(s.get("chunk"))
    if idx is not None:
        btn.button("📚 전체 보기", key=key, on_click=goto_article, args=(idx,),
                   help="왼쪽 '조항 직접 보기'에서 이 조항 원문 전체를 엽니다.")
    st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


def render_sources(sources, grounded=True, answer="", turn_no=0):
    k = lambda j: f"goto_{turn_no}_{j}"
    if not grounded:
        with st.expander("🔎 가장 가까웠던 조항 (근거로 쓰지 않음)"):
            for j, s in enumerate(sources[:1]):
                _source_block(s, k(j))
        return

    cited = cited_articles(answer)
    used = [s for s in sources if s["article"] in cited]
    others = [s for s in sources if s["article"] not in cited]
    if not used:                       # 답변에 조항 표기가 없으면 찾은 순서대로 보여줌
        used, others = sources, []
    with st.expander(f"🔎 참고한 조항 보기 (출처 {len(used)}건)"):
        st.markdown("✅ **답변 근거**")
        for j, s in enumerate(used):
            _source_block(s, k(j))
        if others:
            st.divider()
            st.caption("함께 찾았지만 답변에는 쓰지 않은 조항")
            for j, s in enumerate(others, start=len(used)):
                _source_block(s, k(j))


def render_meta(turn):
    t = turn["timing"]
    if turn["grounded"]:
        st.caption(f"⏱ 응답 {t['total']:.1f}초 (조항 검색 {t['search']:.1f}초 · 답변 생성 {t['gen']:.1f}초)"
                   f" · 근거 일치도 {turn['top_score']:.2f}")
    else:
        st.caption(f"⚠️ 관련 조항 없음 · 근거 일치도 {turn['top_score']:.2f} (기준 {turn['min_score']:.2f} 미만)"
                   f" · ⏱ {t['total']:.1f}초")


def export_text(history, doc_name):
    lines = [f"규정 도우미 대화 기록 — {datetime.now():%Y-%m-%d %H:%M}",
             f"문서: {doc_name}", "=" * 40, ""]
    for i, h in enumerate(history, 1):
        cited = cited_articles(h["answer"])
        srcs = [s for s in h["sources"] if s["article"] in cited] or h["sources"]
        refs = ", ".join(dict.fromkeys(s.get("label") or s["article"] or "본문" for s in srcs))
        lines += [f"[Q{i}] {h['q']}", "", h["answer"], "",
                  f"- 참고 조항: {refs}" if h["grounded"] else "- 관련 조항 없음",
                  f"- 응답 시간: {h['timing']['total']:.1f}초", "-" * 40, ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🔒 규정 도우미")
st.caption("규정 PDF를 자연어로 질문하면, 근거 조항과 함께 답합니다. 모든 처리는 개인 PC 안에서 실행됩니다.")

client = get_client()
connected, installed = list_models(client)

# --- 사이드바 ---
with st.sidebar:
    st.header("설정")

    # 답변 모델 선택 (설치된 모델 중에서)
    gen_options = [n for n in installed if not n.startswith(EMBED_MODEL)]
    if gen_options:
        default_idx = gen_options.index(DEFAULT_GEN_MODEL) if DEFAULT_GEN_MODEL in gen_options else 0
        gen_model = st.selectbox("답변 모델", gen_options, index=default_idx,
                                 help="Ollama에 설치된 모델 중에서 고릅니다.")
    else:
        gen_model = DEFAULT_GEN_MODEL

    # 상태 표시
    st.subheader("상태")
    has_embed = _full_name(EMBED_MODEL) in installed
    has_gen = _full_name(gen_model) in installed
    st.markdown(f"{'✅' if connected else '❌'} Ollama 연결")
    st.markdown(f"{'✅' if has_embed else '❌'} 임베딩 모델 ({EMBED_MODEL})")
    st.markdown(f"{'✅' if has_gen else '❌'} 답변 모델 ({gen_model})")
    ready = connected and has_embed and has_gen

    if ready:
        try:
            with st.spinner("모델 예열 중... (처음 한 번만)"):
                warm_up(gen_model)
            st.markdown("✅ 모델 예열 완료")
        except Exception as e:
            st.markdown(f"⚠️ 예열 실패: {e}")
    elif not connected:
        st.error("Ollama가 실행 중이 아닙니다. 작업표시줄 라마 아이콘을 확인하세요.")
    else:
        missing = [n for n, ok in ((EMBED_MODEL, has_embed), (gen_model, has_gen)) if not ok]
        st.warning("모델을 먼저 받으세요:\n" + "\n".join(f"`ollama pull {n}`" for n in missing))

    # 문서
    st.subheader("문서")
    uploaded = st.file_uploader("규정 PDF 업로드", type=["pdf"])
    if "doc_name" in st.session_state:
        st.markdown(f"📄 **{st.session_state.doc_name}**")
        st.caption(f"{len(st.session_state.articles)}개 조항 · 분석 {st.session_state.index_sec:.1f}초")

        # 조항 직접 보기 ('전체 보기' 버튼으로 이동해 오면 자동으로 펼쳐짐)
        arts = st.session_state.articles
        if st.session_state.get("article_pick", 0) >= len(arts):
            st.session_state.article_pick = 0
        opened = st.session_state.get("article_open", False)
        cur = arts[st.session_state.get("article_pick", 0)][0]
        # 제목에 현재 조항을 넣어, 다른 조항으로 이동하면 새로 펼쳐지도록 함
        with st.expander(f"📚 조항 직접 보기 · {cur}" if opened else "📚 조항 직접 보기", expanded=opened):
            pick = st.selectbox("조항 선택", range(len(arts)),
                                format_func=lambda i: arts[i][0], key="article_pick")
            label, page, body, _ = arts[pick]
            st.caption(f"p.{page}")
            with st.container(height=320):
                st.text(body)

    # 응답 시간 통계 (발표 실측용)
    answered = [h for h in st.session_state.history if h["grounded"]]
    if answered:
        st.subheader("응답 시간")
        c1, c2 = st.columns(2)
        c1.metric("질문 수", f"{len(answered)}건")
        c2.metric("평균", f"{sum(h['timing']['total'] for h in answered) / len(answered):.1f}초")

    # 고급 설정: 근거 판단 기준
    with st.expander("⚙️ 고급 설정"):
        min_score = st.slider(
            "근거 판단 기준 (관련도)", 0.0, 1.0, DEFAULT_MIN_SCORE, 0.01,
            help="가장 관련 높은 조항의 관련도가 이 값보다 낮으면, 답을 지어내지 않고 '규정에서 확인할 수 없음'으로 안내합니다.",
        )

    st.divider()
    if st.session_state.history:
        st.download_button(
            "💾 대화 내용 저장", export_text(st.session_state.history, st.session_state.get("doc_name", "")),
            file_name=f"규정도우미_대화_{datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain", use_container_width=True,
        )
    if st.button("🧹 대화 기록 지우기", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- 문서 불러오기: 업로드 우선, 없으면 기본 취업규칙.pdf 자동 로드 ---
if ready:
    if uploaded is not None:
        data = uploaded.getvalue()
        if st.session_state.get("doc_sig") != hashlib.sha256(data).hexdigest():
            with st.spinner("PDF 분석 중..."):
                if load_document(client, data, uploaded.name):
                    st.rerun()
    elif "doc_name" not in st.session_state and DEFAULT_PDF.exists():
        with st.spinner(f"{DEFAULT_PDF.name} 불러오는 중..."):
            if load_document(client, DEFAULT_PDF.read_bytes(), DEFAULT_PDF.name, is_default=True):
                st.rerun()

# --- 질의응답 ---
if "chunks" in st.session_state and ready:
    st.markdown("**💡 이 문서에 대한 예시 질문 (클릭해서 바로 질문)**")
    clicked = None
    cols = st.columns(2)
    for idx, ex in enumerate(get_examples(client, gen_model)):
        if cols[idx % 2].button(ex, key=f"ex_{idx}", use_container_width=True):
            clicked = ex
    if st.session_state.history:
        st.caption("💬 앞 대화를 이어서 질문할 수 있어요. 예: \"그럼 3년 근무하면요?\"")
    st.divider()

    moved_to = st.session_state.pop("goto_toast", None)
    if moved_to:
        st.toast(f"📚 왼쪽 '조항 직접 보기'에서 {moved_to}을(를) 열었어요.")

    for n, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_meta(turn)
            render_sources(turn["sources"], turn["grounded"], turn["answer"], turn_no=n)

    typed = st.chat_input("질문을 입력하세요")
    question = clicked or typed

    if question:
        history = st.session_state.history
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            try:
                hits, top_score = search(client, question, history, st.session_state.matrix,
                                         st.session_state.get("kw_index"))
            except Exception as e:
                st.error(f"임베딩 실패: {e}")
                st.stop()
            retrieved = [st.session_state.chunks[i] for i, _ in hits]
            grounded = bool(hits) and top_score >= min_score
            t1 = time.perf_counter()

            if grounded:
                try:
                    answer = st.write_stream(
                        stream_answer(client, gen_model, build_messages(question, retrieved, history)))
                except Exception as e:
                    st.error(f"답변 생성 실패: {e}\n'ollama pull {gen_model}'를 실행했는지 확인하세요.")
                    st.stop()
            else:
                answer = NO_GROUNDS_MSG
                st.write(answer)
            t2 = time.perf_counter()

            history.append({
                "q": question,
                "answer": answer,
                "grounded": grounded,
                "top_score": top_score,
                "min_score": min_score,
                "timing": {"search": t1 - t0, "gen": t2 - t1, "total": t2 - t0},
                "sources": [
                    {"article": c["article"], "label": article_label(c), "page": c["page"],
                     "score": s, "text": c["text"], "chunk": i}
                    for (i, s), c in zip(hits, retrieved)
                ],
            })
            st.rerun()  # 기록·통계를 갱신해 다시 그림
elif ready:
    st.info(f"👈 사이드바에서 규정 PDF를 올리거나, 이 파일과 같은 폴더에 '{DEFAULT_PDF.name}'를 넣어두세요.")
else:
    st.info("👈 사이드바 '상태'에서 준비가 안 된 항목을 먼저 해결하세요.")
