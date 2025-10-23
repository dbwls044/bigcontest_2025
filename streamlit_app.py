import streamlit as st
import pandas as pd
import google.generativeai as genai
from textwrap import dedent
from pathlib import Path
from PIL import Image
import re
import streamlit as st

# ------------------------------
# 기본 설정
# ------------------------------
st.set_page_config(page_title="AI 전략 참모", layout="wide")
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ------------------------------
# 상단 헤더 구성
# ------------------------------
ASSETS = Path("assets")
logo_path = ASSETS / "shc_ci_basic_00.png"

col1, col2 = st.columns([2, 5])
with col1:
    st.image(str(logo_path), width=250)
with col2:
    st.markdown(
        "<h4 style='text-align:right; color:#444;'>2025 Big Contest · AI DATA 활용분야</h4>",
        unsafe_allow_html=True
    )

st.markdown("---") 

# ------------------------------
# 결과 출력 디자인 함수
# ------------------------------
def format_report(text: str) -> str:
    replacements = {
        r"\[진단 및 인사이트\]": "<h3 style='color:#16A085'>🔍 진단 및 인사이트</h3>",
        r"\[3 Key Insights[^\]]*\]": "<h3 style='color:#16A085'>🔍 진단 및 인사이트</h3>",
        r"\[우리 매장을 위한 마케팅 아이디어\]": "<h3 style='color:#D35400'>💡 우리 매장을 위한 마케팅 아이디어</h3>",
        r"\[3 Marketing Ideas[^\]]*\]": "<h3 style='color:#D35400'>💡 우리 매장을 위한 마케팅 아이디어</h3>",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

# ------------------------------
# Gemini 설정
# ------------------------------
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ------------------------------
# 제목
# ------------------------------
st.title("🤖 AI 전략 참모 - 데이터 기반 의사결정 지원")
st.subheader("내 가게를 살리는 AI 비밀 상담사, 찰떡 마케팅 전략을 찾아라!")

# ------------------------------
# 데이터 로드
# ------------------------------
DATA_PATH = Path("data/bigcon_data.csv")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"데이터 파일이 없습니다: {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

data = load_data()

# ------------------------------
# System Prompt
# ------------------------------
system_prompt = dedent("""
You are the AI Business Strategy Advisor for small merchants.
Base your analysis on the uploaded dataset, but if some data is missing, make reasonable assumptions.
Do not repeat that information is missing — always provide meaningful analysis and marketing ideas.
Be clear and concise.

EXPLAINED_COLUMNS   
- MCT_OPE_MS_CN: 가맹점 운영개월수 구간 (6개 구간: 10%이하~90%초과). 작을수록 상위(오래 운영)
- RC_M1_SAA: 매출금액 구간 (6개 구간). 작을수록 상위(매출금액 상위)
- RC_M1_TO_UE_CT: 매출건수 구간 (6개 구간). 작을수록 상위(매출건수 상위)
- RC_M1_UE_CUS_CN: 유니크 고객 수 구간 (6개 구간). 작을수록 상위(고객 수 상위)
- RC_M1_AV_NP_AT: 객단가 구간 (6개 구간). 작을수록 상위(객단가 상위)
- APV_CE_RAT: 취소율 구간 (6개 구간). 1구간에 가까울수록 상위(취소율 낮음)/6구간에 가까울수록 취소율 높음
- DLV_SAA_RAT: 배달매출금액 비율 (배달매출 미존재시 SV=-999999.9)
- M1_SME_RY_SAA_RAT: 동일 업종 매출금액 비율 (업종 평균 대비, 평균=100%)
- M1_SME_RY_CNT_RAT: 동일 업종 매출건수 비율 (업종 평균 대비, 평균=100%)
- M12_SME_RY_SAA_PCE_RT: 동일 업종 내 매출 순위 비율 (업종 내 순위/전체*100, 작을수록 상위)
- M12_SME_BZN_SAA_PCE_RT: 동일 상권 내 매출 순위 비율 (상권 내 순위/전체*100, 작을수록 상위)
- M12_SME_RY_ME_MCT_RAT: 동일 업종 내 해지 가맹점 비중
- M12_SME_BZN_ME_MCT_RA: 동일 상권 내 해지 가맹점 비중
""").strip()

# ------------------------------
# Scenario Prompt
# ------------------------------
scenario_prompts = {
    "카페업종": """
    당신은 카페 마케팅 전문가입니다.
    주어진 데이터를 바탕으로 이 카페의 주요 방문 고객 특성을 분석하고,
    고객 특성에 맞는 마케팅 채널(예: 인스타그램, 네이버 블로그, 쿠폰 앱 등)을 추천하세요.
    마지막에는 실제 홍보 문구 예시도 작성하세요.
    """,

    "재방문율 30% 이하": """
    당신은 고객 유지 전문가입니다.
    주어진 데이터를 바탕으로 매장의 재방문율이 낮은 원인을 진단하고,
    재방문을 높일 수 있는 마케팅 아이디어와 근거를 제시하세요.
    행동 가능한 제안 2가지를 표 형태로 요약하세요.
    """,

    "요식업종": """
    당신은 요식업 전문 마케팅 컨설턴트입니다.
    주어진 데이터를 바탕으로 현재 매장의 가장 큰 문제점(매출, 고객, 외부요인 중)을 찾아내고,
    이를 개선하기 위한 구체적 마케팅 아이디어와 근거를 제시하세요.
    실제 실행 가능한 전략 2~3개를 제시하세요.
    """,

    "기타": """
    주어진 데이터를 바탕으로, 매출과 고객 패턴을 분석하고 일반적인 마케팅 전략을 제시하세요.
    """,

    "기상데이터 기반 전략": """
    당신은 날씨 데이터를 활용한 마케팅 전문가입니다.  
    아래 데이터를 기반으로 평균기온, 월합강수량, 최심적설의 기상 요인이  
    매출이나 방문 고객 수에 어떤 영향을 주었는지를 분석하세요.  

    특히 다음 포인트에 주목하세요:
    - 기온 상승/하강에 따른 업종별 매출 및 방문 패턴의 변화
    - 강수량 또는 폭설 시 방문 감소 혹은 배달 매출 비중의 변화
    - 계절적 요인(예: 여름 vs 겨울)과 프로모션의 연결 가능성

    결과적으로 다음을 제시하세요:
    1. 날씨 요인과 매출/고객 변화 간의 핵심 상관관계 2~3가지  
    2. 해당 매장(또는 업종)에 적합한 기상 연동형 마케팅 전략 제안  
    (예: 비 오는 날 할인, 추운 날 테이크아웃 프로모션 등)  
    3. 실제 적용 가능한 구체적 행동 방안

    결과는 다음 섹션 형태로 출력하세요:

    ### 🔍 진단 및 인사이트  
    - 날씨와 매출 간 주요 관계 요약 (수치/패턴 중심)  
    - 고객 행동 변화의 원인 분석  

    ### 💡 기상 연동형 마케팅 전략  
    - 전략 이름  
    - 실행 방법  
    - 기대 효과
    """,

    "주요고객층 기반 전략": """
    당신은 고객 세분화 데이터를 활용하는 마케팅 컨설턴트입니다.  
    아래 데이터를 통해 주요 고객층(성별/연령대)에 따른 소비 패턴과  
    매출 기여도를 분석하고, 각 그룹별로 차별화된 마케팅 전략을 제안하세요.

    다음 포인트에 주목하세요:
    - 남성/여성 고객 간 결제 빈도 또는 객단가 차이
    - 10~20대, 30대, 40대, 50대 이상 연령대별 매출 비중 변화
    - 특정 연령층이 많이 방문하는 시기나 요일, 상품 카테고리 특징

    결과적으로 다음을 제시하세요:
    1. 주요 고객층의 매출/방문 패턴 요약 (가장 비중이 큰 집단 중심)
    2. 연령·성별별 특화 마케팅 아이디어 (예: 20대 여성 중심 SNS 이벤트)
    3. 고객층 확장을 위한 보조 타깃 제안 (ex. 40대 남성 신규 유입 전략)

    결과는 다음 섹션 형태로 출력하세요:

    ### 🔍 진단 및 인사이트  
    - 성별/연령별 매출·방문 비율의 주요 차이  
    - 주요 고객층의 소비 특성과 트렌드  

    ### 💡 고객 세분화 기반 마케팅 전략  
    - 핵심 타깃 전략  
    - 실행 채널 (SNS, 쿠폰, 오프라인 이벤트 등)  
    - 기대 효과
    """
}

# ------------------------------
# Streamlit 로직
# ------------------------------
if data is not None:
    st.success("✅ 데이터 로드 완료!")

    # 표시용 매장명
    data["DISPLAY_NAME"] = (
        data["HPSN_MCT_BZN_CD_NM"].astype(str).str.strip() + " · " +
        data["HPSN_MCT_ZCD_NM"].astype(str).str.strip() + " · " +
        data["MCT_NM"].astype(str).str.strip() + " · " +
        data["ENCODED_MCT"].astype(str).str.strip()
    )

    st.markdown("### 🔍 가맹점코드로 검색하기")
    code_input = st.text_input("가맹점구분번호(ENCODED_MCT)를 입력하세요", placeholder="예: MCT01234")

    if code_input:
        code_input = code_input.strip()
        store_data = data[data["ENCODED_MCT"].astype(str).str.strip() == code_input]

        if store_data.empty:
            st.error("❌ 해당 가맹점코드로 등록된 매장이 없습니다.")
        else:
            selected_display = store_data["DISPLAY_NAME"].iloc[0]
            st.subheader(f"🏪 선택된 가맹점: {selected_display}")

            # --- 매출금액구간, 매출건수구간, 재방문 고객 비중, 신규 고객 비중 ---
            store_data = store_data.copy()
            store_data.loc[:, "RC_M1_SAA_num"] = store_data["RC_M1_SAA"].astype(str).str.extract(r"(\d+)").astype(float)
            store_data.loc[:, "RC_M1_TO_UE_C_num"] = store_data["RC_M1_TO_UE_CT"].astype(str).str.extract(r"(\d+)").astype(float)

            sales_amount_section = int(round(store_data["RC_M1_SAA_num"].mean()))
            sales_count_section = int(round(store_data["RC_M1_TO_UE_C_num"].mean()))
            revisit_rate = store_data["MCT_UE_CLN_REU_RAT"].mean()
            new_customer_rate = store_data["MCT_UE_CLN_NEW_RAT"].mean()

            # --- 주요고객층(성별/연령) ---
            age_gender_cols = [
                "M12_MAL_1020_RAT", "M12_MAL_30_RAT", "M12_MAL_40_RAT", "M12_MAL_50_RAT", "M12_MAL_60_RAT",
                "M12_FME_1020_RAT", "M12_FME_30_RAT", "M12_FME_40_RAT", "M12_FME_50_RAT", "M12_FME_60_RAT"
            ]
            if any(col in store_data.columns for col in age_gender_cols):
                true_counts = store_data[age_gender_cols]
                main_age_gender_col = true_counts.idxmax(axis=1).iloc[0]
                main_customer_age_gender = (
                    main_age_gender_col.replace("주고객층_", "")
                    .replace("_RAT", "")
                    .replace("M12_", "")
                    .replace("_", " ")
                    .replace("FME", "여성")
                    .replace("MAL", "남성")
                    .replace("1020", "10~20대")
                    .replace("30", "30대")
                    .replace("40", "40대")
                    .replace("50", "50대")
                    .replace("60", "60대 이상")
                    .strip()
                )
            else:
                main_customer_age_gender = "데이터 없음"

            # --- 주요 고객층(유형) ---
            type_cols = ["RC_M1_SHC_RSD_UE_CLN_RAT", "RC_M1_SHC_WP_UE_CLN_RAT", "RC_M1_SHC_FLP_UE_CLN_RAT"]
            if any(col in store_data.columns for col in type_cols):
                type_means = store_data[type_cols].mean()
                main_type_col = type_means.idxmax()
                main_customer_type = (
                    main_type_col.replace("RC_M1_SHC_", "")
                                .replace("_UE_CLN_RAT", "")
                                .replace("RSD", "거주 이용 고객")
                                .replace("WP", "직장 이용 고객")
                                .replace("FLP", "유동인구 이용 고객")
                                .strip()
                )
            else:
                main_customer_type = "데이터 없음"

            # --- 주요 지표 표 ---
            st.markdown("#### 📊 주요 지표 요약")
            summary_df = pd.DataFrame({
                "항목": ["매출금액구간(작을수록 상위)", "매출건수구간(작을수록 상위)", "재방문 고객 비중", "신규 고객 비중", "주요 고객층(성별/연령)", "주요 고객층(유형)"],
                "값": [f"{sales_amount_section}구간", 
                    f"{sales_count_section}구간", 
                    f"{revisit_rate:.1f}%", 
                    f"{new_customer_rate:.1f}%", 
                    main_customer_age_gender, 
                    main_customer_type]
            })
            st.table(summary_df)

            # ------------------------------
            #  점주 고민 입력
            # ------------------------------
            greeting = "💬 AI 전략 참모가 데이터를 분석할 준비가 되었습니다.\n점주님의 고민을 입력해보세요!"
            with st.chat_message("assistant"):
                st.write(greeting)

            user_question = st.chat_input("💭 점주님의 고민을 입력하세요", key="user_input") or ""

            # --- 시나리오 자동 선택 로직 ---
            if "카페" in selected_display or "커피" in selected_display :
                scenario = "카페업종"
            elif revisit_rate < 30:
                scenario = "재방문율 30% 이하"
            elif any(keyword in selected_display for keyword in ["가정식", "분식", "한식", "양식", "중식", "일식"]):
                scenario = "요식업종"
            elif any(word in user_question for word in ["날씨", "비", "눈", "기온", "계절"]):
                scenario = "기상데이터 기반 전략"
            else:
                scenario = "기타"

            scenario_prompt = scenario_prompts[scenario]

            if user_question:
                st.markdown(f"### 💭 {user_question}")

                def safe_generate(prompt, retries=2, timeout=60):
                    import time
                    for attempt in range(retries):
                        start = time.time()
                        try:
                            response = model.generate_content(
                                contents=[prompt],
                                generation_config={
                                    "temperature": 0.3,
                                    "max_output_tokens": 6144,
                                    "candidate_count": 1,
                                    "top_p": 0.8,
                                },
                            )
                            if time.time() - start > timeout:
                                raise TimeoutError

                            if hasattr(response, "candidates") and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, "content") and candidate.content.parts:
                                    return candidate.content.parts[0].text.strip()
                                else:
                                    return "⚠️ 분석 결과가 비어 있습니다. 프롬프트가 너무 길거나 응답이 중단되었습니다."
                            else:
                                return "⚠️ 모델이 유효한 응답을 반환하지 않았습니다."

                        except Exception as e:
                            if attempt < retries - 1:
                                time.sleep(2)
                                continue
                            else:
                                return f"⚠️ 분석 중 오류 발생: {str(e)}"

                with st.spinner("🔍 AI 전략 참모가 데이터를 분석 중입니다..."):
                    
                    # 1단계: 인사이트 생성
                    st.markdown("#### 1단계: 인사이트 분석 중...")
                    insight_placeholder = st.empty()

                    insight_prompt = f"""
                    You are a data analyst helping a small business owner understand store performance.

                    Focus on identifying:
                    1. Key patterns or anomalies in sales and customer behavior
                    2. Possible reasons for low sales/customers
                    3. 2~3 actionable insights that explain the current business state

                    [Store Name]
                    {selected_display}

                    [Sample Data]
                    {store_data.sample(n=min(5, len(store_data)), random_state=42).to_string(index=False)}

                    [Summary Statistics]
                    {store_data.describe().round(2).to_string(index=False)}

                    The owner asked: "{user_question}"

                    Keep it clear, data-driven, and concise.

                    Make sure the entire response is written in fluent, natural Korean language.
                    """

                    insight_text = safe_generate(insight_prompt)

                    if insight_text and "⚠️" not in insight_text:
                        insight_placeholder.markdown("✅ 인사이트 분석 완료!")
                        st.markdown("### 🔍 진단 및 인사이트")
                        st.markdown(format_report(insight_text), unsafe_allow_html=True)
                    else:
                        insight_placeholder.markdown("❌ 인사이트 분석 실패")
                        st.stop()

                    # 2단계: 전략 제안 생성
                    st.markdown("#### 2단계: 마케팅 전략 생성 중...")
                    strategy_placeholder = st.empty()

                    strategy_prompt = f"""
                    You are a marketing strategist for small merchants.

                    Based on the insights below, write 2~3 realistic marketing strategies 
                    to increase revisit rate and improve customer retention.

                    Each strategy should include:
                    - 🎯 Goal (what problem it addresses)
                    - 💡 Action (what to do)
                    - 📈 Expected Effect (why it works)

                    [Store Insights]
                    {insight_text}

                    [Owner's Question]
                    {user_question}

                    Keep it clear, data-driven, and concise.

                    Make sure the entire response is written in fluent, natural Korean language.
                    """

                    strategy_text = safe_generate(strategy_prompt)

                    
                    if strategy_text and "⚠️" not in strategy_text:
                        strategy_placeholder.markdown("✅ 전략 제안 완료!")
                        st.markdown("### 💡 우리 매장을 위한 마케팅 전략")
                        st.markdown(format_report(strategy_text), unsafe_allow_html=True)
                    else:
                        strategy_placeholder.markdown("❌ 전략 생성 실패")

                