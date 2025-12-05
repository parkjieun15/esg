# app.py

import os
import json
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------------------------------
# 0. OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 사용)
# ---------------------------------------
client = OpenAI()


# ---------------------------------------
# 1. 데이터 로딩
# ---------------------------------------
@st.cache_data
def load_data(csv_path: str = "cache_merged_esg.csv"):
    df = pd.read_csv(csv_path)

    # 날짜 컬럼 dt → datetime 변환
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    else:
        st.error("dt 컬럼을 찾을 수 없습니다. CSV에 dt(날짜) 컬럼이 있는지 확인하세요.")

    # 임베딩용 텍스트: clean_text → summary → title 순으로 보완
    df["text_for_embed"] = (
        df.get("clean_text", "")
        .fillna(df.get("summary", ""))
        .fillna(df.get("title", ""))
        .fillna("")
    )

    # 결측값 최소화
    if "esg_tag" not in df.columns:
        df["esg_tag"] = "unknown"
    if "severity_ai" not in df.columns:
        df["severity_ai"] = 0.0
    if "tone" not in df.columns:
        df["tone"] = "unknown"

    return df


# ---------------------------------------
# 2. 임베딩 모델 + FAISS 인덱스 빌드 (캐시)
# ---------------------------------------
@st.cache_resource
def build_embed_and_index(texts, cache_dir="faiss_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    index_path = os.path.join(cache_dir, "faiss.index")

    # 1) 캐시가 있으면: 바로 로드
    if os.path.exists(emb_path) and os.path.exists(index_path):
        embs = np.load(emb_path).astype("float32")
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        # 모델은 매번 로드 (비교적 가벼움)
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return model, embs, index

    # 2) 캐시가 없으면: 처음 한 번만 계산
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # 디스크에 저장
    np.save(emb_path, embs)
    faiss.write_index(index, index_path)

    return model, embs, index


# ---------------------------------------
# 3. 의미기반 검색 함수 (FAISS)
# ---------------------------------------
def semantic_search(query: str, model, index, df, top_k: int = 30):
    # 질의가 비어 있으면 그냥 원본 df 반환
    if not query.strip():
        return df.copy()

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    scores = D[0]
    idxs = I[0]

    rows = []
    for score, idx in zip(scores, idxs):
        row = df.iloc[int(idx)].copy()
        row["similarity"] = float(score)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["similarity"])

    result = pd.DataFrame(rows)
    return result


# ---------------------------------------
# 4. 질의 파싱 (기업/기간/키워드/의도 추출)
# ---------------------------------------
def parse_user_query(query: str):
    """
    사용자 자연어 질의를 JSON 구조로 파싱.
    company, period, esg_keywords, intent 등을 추출.
    """
    if not query.strip():
        # 비어 있으면 기본값
        return {
            "companies": [],
            "start_date": None,
            "end_date": None,
            "esg_keywords": [],
            "intent": "summary",
        }

    system_prompt = (
        "너는 한국어 금융·ESG 애널리스트 어시스턴트다. "
        "사용자의 질문에서 기업명, 기간, ESG 관련 키워드, 사용 의도를 구조화하여 JSON으로만 출력하라."
    )

    user_prompt = f"""
질문: {query}

다음 JSON 형식으로만 답해라 (설명 문장 절대 쓰지 말 것).

{{
  "companies": ["기업명1", "기업명2"],   // 없으면 빈 배열
  "start_date": "YYYY-MM-DD" 또는 null,   // '작년', '최근 6개월' 등은 적당히 해석
  "end_date": "YYYY-MM-DD" 또는 null,
  "esg_keywords": ["탄소배출", "환경오염", ...],  // 관련 키워드 3~7개
  "intent": "summary" 또는 "risk_focus" 또는 "comparison" 또는 "other"
}}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        # 파싱 실패 시 안전한 기본값
        data = {
            "companies": [],
            "start_date": None,
            "end_date": None,
            "esg_keywords": [],
            "intent": "summary",
        }
    return data


# ---------------------------------------
# 5. 개별 기사에 대한 ESG 정량 평가
# ---------------------------------------
def llm_score_document(row: pd.Series, query_info: dict):
    """
    단일 기사에 대해:
    - ESG 분류(E/S/G/mixed/none)
    - E/S/G별 점수 (0~3)
    - total_severity (0~9)
    - key_sentences (2~3개)
    를 JSON으로 받아온다.
    """
    title = row.get("title", "")
    text = row.get("clean_text", row.get("summary", ""))
    date_str = str(row.get("dt", ""))[:10]

    qi = query_info

    scoring_rules = """
[심각도 계산 규칙]

각 점수를 0~3 사이 정수로 부여한다. 기준은 다음과 같다.

0점: 관련 리스크 거의 없음 또는 단순 홍보/일반 소식
1점: 잠재적/경미한 리스크 (회사 실적이나 평판에 큰 영향은 적음)
2점: 중간 수준의 리스크 (규제, 평판, 비용 증가 등 가시적 영향 가능)
3점: 매우 큰 리스크 (법적 제재, 대규모 사고, ESG 등급 강등 가능 수준)

- E_score: 환경(탄소배출, 오염, 에너지, 자원, 기후) 관련 리스크 정도
- S_score: 사회(안전사고, 인권, 지역사회, 노동, 고객/협력사) 리스크 정도
- G_score: 지배구조(이사회, 오너리스크, 회계부정, 내부통제) 리스크 정도

total_severity = E_score + S_score + G_score (0~9)
"""

    prompt = f"""
너는 기관투자자 대상 한국 기업 ESG 리스크 분석가다.

[사용자 질의 정보]
- 회사 후보: {qi.get("companies", [])}
- ESG 키워드: {qi.get("esg_keywords", [])}
- 의도: {qi.get("intent", "")}

[기사 정보]
- 날짜: {date_str}
- 제목: {title}
- 본문: {text}

위 기사 내용만을 기준으로 ESG 리스크를 평가하라.

{scoring_rules}

다음 JSON 형식으로만 답해라 (설명 문장 쓰지 말 것).

{{
  "esg_category": "E" 또는 "S" 또는 "G" 또는 "mixed" 또는 "none",
  "E_score": 0~3 정수,
  "S_score": 0~3 정수,
  "G_score": 0~3 정수,
  "total_severity": 0~9 정수,
  "key_sentences": ["기사에서 인용한 중요한 문장1", "문장2", "문장3"],
  "reason": "점수를 이렇게 준 이유를 2~3문장으로 요약"
}}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {
            "esg_category": "none",
            "E_score": 0,
            "S_score": 0,
            "G_score": 0,
            "total_severity": 0,
            "key_sentences": [],
            "reason": "",
        }

    return data

def llm_summarize_top_docs(top_docs: pd.DataFrame, query_info: dict):
    """
    TOP-10 문서 전체를 하나의 ESG 리포트로 요약하는 LLM 호출.
    """
    docs_context = []

    for _, row in top_docs.iterrows():
        # 날짜 처리
        if pd.notna(row.get("dt")):
            try:
                date_str = str(row["dt"].date())
            except Exception:
                date_str = str(row.get("dt"))
        else:
            date_str = ""

        # 요약용 텍스트 처리 (항상 문자열로 변환)
        text_src = row.get("clean_text")
        if not isinstance(text_src, str):
            text_src = row.get("summary")
        if not isinstance(text_src, str):
            text_src = ""
        text_src = str(text_src)
        text_snippet = text_src[:400]

        # severity_final (NaN → 0)
        sev_val = row.get("severity_final", 0.0)
        try:
            sev_val = float(sev_val)
        except Exception:
            sev_val = 0.0

        docs_context.append(
            {
                "date": date_str,
                "title": row.get("title", ""),
                "esg_category": row.get("esg_llm", "none"),
                "severity_final": sev_val,
                "summary": text_snippet,
                "url": row.get("url", ""),
            }
        )

    # LLM 입력용 JSON
    context_json = json.dumps(docs_context, ensure_ascii=False, indent=2)
    qi_str = json.dumps(query_info, ensure_ascii=False)

    prompt = f"""
너는 한국 대형 기관투자자의 ESG 애널리스트이다.

[사용자 질의 정보]
{qi_str}

[후보 문서 10개 요약 정보]
{context_json}

개별 기사 설명은 하지 말고,
'전체적으로 어떤 ESG 이슈와 리스크가 중요한지'만 하나의 리포트로 정리하라.

다음 JSON 형식으로만 출력:
{{
  "overall_comment": "핵심 요약 3~5문장",
  "overall_key_sentences": ["중요 근거 문장 3~6개"],
  "risk_comment": "정량 심각도(severity_final 평균)를 기반으로 투자자 시사점 2~3문장",
  "esg_focus": "E 또는 S 또는 G 또는 mixed",
  "representative_urls": ["중요 참고용 URL 3~5개"]
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {
            "overall_comment": "",
            "overall_key_sentences": [],
            "risk_comment": "",
            "esg_focus": "mixed",
            "representative_urls": [],
        }

    return data

# ---------------------------------------
# 6. Streamlit UI
# ---------------------------------------
def main():
    st.set_page_config(
        page_title="POSCO ESG 뉴스 에이전트",
        layout="wide",
    )

    st.title("📊 POSCO ESG 뉴스 에이전트")

    # 6-1) 데이터 로딩 & 임베딩 인덱스 준비
    df = load_data()
    st.caption(f"데이터 행 개수: {len(df):,}건")

    with st.spinner("임베딩 및 검색 인덱스를 준비하는 중입니다..."):
        embed_model, embeddings, index = build_embed_and_index(
            df["text_for_embed"].tolist()
        )

    # ---------------------------
    # SIDEBAR - 자연어 질의만 사용
    # ---------------------------
    st.sidebar.header("검색 / 필터 설정")

    # 🔎 자연어 질문 입력란
    user_query = st.sidebar.text_input(
        "자연어로 질문해보세요 (예: '최근 POSCO 탄소배출 관련 리스크 정리해줘')",
        value="최근 환경 리스크 요약해줘",
    )

    # 검색 실행 버튼
    run_button = st.sidebar.button("🔍 검색 및 분석 실행")

    # -----------------------------------
    # MAIN 영역 - 결과 출력
    # -----------------------------------
    if not run_button:
        st.info("왼쪽에서 질문을 입력한 뒤 **[🔍 검색 및 분석 실행]** 버튼을 눌러주세요.")
        return

    # 1) 의미 기반 검색 (FAISS, 고정 Top-K 후보)
    TOP_K = 50  # 후보 문서 50개 중에서 최종 TOP-10 뽑기
    with st.spinner("의미 기반 검색 중..."):
        search_df = semantic_search(user_query, embed_model, index, df, top_k=TOP_K)

    if search_df.empty:
        st.warning("검색 결과가 없습니다. 질문을 조금 다르게 써보거나 데이터를 확인해보세요.")
        return

    # 2) 사용자 질의 파싱 → 기업명 / 기간 / 키워드 / 의도 등 추출
    with st.spinner("질의 내용을 분석하는 중입니다..."):
        query_info = parse_user_query(user_query)

    filtered = search_df.copy()

    # 2-1) 자연어에서 파악한 기간으로 필터링 (start_date / end_date)
    if "dt" in filtered.columns:
        if query_info.get("start_date"):
            start_date = pd.to_datetime(query_info["start_date"]).date()
            filtered = filtered[filtered["dt"].dt.date >= start_date]
        if query_info.get("end_date"):
            end_date = pd.to_datetime(query_info["end_date"]).date()
            filtered = filtered[filtered["dt"].dt.date <= end_date]

    if filtered.empty:
        st.warning("질의에서 추출한 기간 조건을 적용하니 남는 문서가 없습니다. 질문을 조금 더 넓게 적어보세요.")
        return

    # -----------------------------
    # 3) TOP-10 문서 선정 및 LLM 정량 분석
    # -----------------------------
    # similarity 높은 순으로 정렬 후 상위 10개만 사용
    if "similarity" in filtered.columns:
        filtered = filtered.sort_values("similarity", ascending=False)

    top_docs = filtered.head(10).copy().reset_index(drop=True)

    # 3-2) 각 문서별 ESG 분류 + 정량 심각도 + 근거문장
    results = []
    with st.spinner("선택된 기사들에 대해 ESG 분류 및 정량 심각도 평가 중입니다..."):
        for i, row in top_docs.iterrows():
            r = llm_score_document(row, query_info)
            results.append(r)

    # 결과를 top_docs에 붙이기
    top_docs["esg_llm"] = [r["esg_category"] for r in results]
    top_docs["E_score"] = [r["E_score"] for r in results]
    top_docs["S_score"] = [r["S_score"] for r in results]
    top_docs["G_score"] = [r["G_score"] for r in results]
    top_docs["severity_llm"] = [r["total_severity"] for r in results]
    top_docs["llm_reason"] = [r["reason"] for r in results]
    top_docs["llm_key_sentences"] = [r["key_sentences"] for r in results]

    # -----------------------------
    # 4) 관련도·최신성·위험도의 **가중합**으로 최종 심각도 계산
    # -----------------------------

    # (1) 관련도: similarity → 0~1 정규화
    rel_raw = top_docs["similarity"].fillna(0.0)
    rel_min, rel_max = rel_raw.min(), rel_raw.max()
    if rel_max > rel_min:
        relevance = (rel_raw - rel_min) / (rel_max - rel_min)
    else:
        relevance = pd.Series(0.5, index=top_docs.index)  # 모두 같으면 중간값

    # (2) 위험도: LLM total_severity (0~9) → 0~1
    risk = top_docs["severity_llm"].fillna(0.0) / 9.0
    risk = risk.clip(lower=0.0, upper=1.0)

    # (3) 최신성: 오늘 기준 날짜 차이 → 0~1
    today = pd.Timestamp.today().normalize()
    if "dt" in top_docs.columns and top_docs["dt"].notna().any():
        age_days = (today - top_docs["dt"].dt.normalize()).dt.days
        age_days = age_days.clip(lower=0)
        max_age = max(age_days.max(), 1)
        recency = 1.0 - (age_days / max_age)
        recency = recency.clip(lower=0.0, upper=1.0)
        # 날짜가 NaT인 경우 0.5로 대체
        recency = recency.fillna(0.5)
    else:
        recency = pd.Series(0.5, index=top_docs.index)

    # (4) 가중치 설정
    # 관련도 0.3, 최신성 0.2, 위험도 0.5  → 합이 1.0
    W_REL = 0.3
    W_REC = 0.2
    W_RISK = 0.5

    # (5) 0~1 범위의 가중합 점수
    score_01 = (
        W_REL * relevance +
        W_REC * recency +
        W_RISK * risk
    )

    # (6) 최종 심각도: 0~10 스케일로 변환
    top_docs["relevance_factor"] = relevance
    top_docs["risk_factor"] = risk
    top_docs["recency_factor"] = recency
    top_docs["severity_final"] = 10.0 * score_01

    # -----------------------------
    # 5) 날짜별 평균 심각도 그래프 (6번: 그래프)
    # -----------------------------
    st.subheader("📈 선택된 문서들의 날짜별 평균 심각도(severity_final) 추이")

    trend_df = (
        top_docs.dropna(subset=["dt"])
        .assign(date=lambda x: x["dt"].dt.date)
        .groupby("date")["severity_final"]
        .mean()
        .sort_index()
    )

    if not trend_df.empty:
        st.line_chart(trend_df)
    else:
        st.write("그래프를 그릴 수 있는 데이터가 충분하지 않습니다.")

    # -----------------------------
    # -----------------------------
    # 6) TOP-10 통합 리포트 (하나만 보여줌)
    # -----------------------------
    # 숫자 요약값들 먼저 계산
    avg_E = top_docs["E_score"].mean()
    avg_S = top_docs["S_score"].mean()
    avg_G = top_docs["G_score"].mean()
    overall_sev = top_docs["severity_final"].mean()

    with st.spinner("TOP-10 문서를 통합 분석하는 중입니다..."):
        summary = llm_summarize_top_docs(top_docs, query_info)

    esg_focus = summary.get("esg_focus", "mixed")
    overall_comment = summary.get("overall_comment", "")
    key_sents = summary.get("overall_key_sentences", []) or []
    risk_comment = summary.get("risk_comment", "")
    rep_urls = summary.get("representative_urls", []) or []

    # 대표 URL이 거의 없으면, top_docs에서 상위 5개 URL을 채워넣기
    if not rep_urls:
        rep_urls = [
            u for u in top_docs["url"].dropna().unique().tolist()
            if isinstance(u, str) and u
        ][:5]

    st.subheader("📘 TOP-10 통합 ESG 리포트")

    col1, col2 = st.columns([3, 1])

    with col1:
        # 1) ESG 분류 결과
        st.markdown("**1) ESG 분류 결과**")
        st.write(f"- LLM이 판단한 주요 초점 영역: {esg_focus}")
        st.write(
            f"- Top10 평균 E/S/G 점수: "
            f"E={avg_E:.2f}, S={avg_S:.2f}, G={avg_G:.2f}"
        )

        # 2) 정량 심각도 (가중합)
        st.markdown("**2) 정량 심각도 (가중합)**")
        st.write(
            "- 관련도·최신성·위험도를 가중합(0.3/0.2/0.5)한 값의 "
            f"평균입니다.\n- Top10 평균 통합 심각도(severity_final): "
            f"**{overall_sev:.2f} / 10**"
        )
        if risk_comment:
            st.write(risk_comment)

        # 3) 요약 / 코멘트
        st.markdown("**3) 요약 / 코멘트**")
        st.write(overall_comment)

        # 4) 근거 문장
        st.markdown("**4) 근거 문장 (key sentences)**")
        if key_sents:
            for s in key_sents:
                st.write(f"- {s}")
        else:
            st.write("- (LLM이 근거 문장을 별도로 제시하지 않았습니다.)")

    with col2:
        # 5) 출처 링크
        st.markdown("**5) 대표 출처 링크**")
        if rep_urls:
            for u in rep_urls:
                st.markdown(f"- [링크]({u})")
        else:
            st.write("- (대표 링크가 없습니다.)")

if __name__ == "__main__":
    main()
