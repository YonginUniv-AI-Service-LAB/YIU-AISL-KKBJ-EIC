import os
import google.genai as genai
from dotenv import load_dotenv

from cost_function import optimize_T, ETC, failure_prob, calc_lcc_savings
from data_loader import get_aadt, get_freeze_thaw
from gp_estimator import gp_estimate, estimate_weibull_params

import numpy as np

load_dotenv()
client = genai.Client(api_key="AIzaSyCKXJ4a53MC4L9M3B_e8-zBDx0RPc871vQ")

REGIONS = [
    "서울", "강원", "제주", "부산", "대구", "인천",
    "광주", "대전", "울산", "경기", "충북", "충남",
    "전북", "전남", "경북", "경남", "세종",
]

DEFAULT_INSPECTION = {
    "years":  [0, 5, 10, 15, 20, 25, 30],
    "grades": [5.0, 4.8, 4.5, 4.1, 3.7, 3.3, 3.0],
}

def extract_region_with_llm(user_input: str) -> str | None:
    prompt = f"""다음 질문에서 교량이 위치한 한국 행정구역(시·도) 이름 하나만 추출하세요.
교량 이름이 나와도 해당 교량이 위치한 시·도를 답하세요. (예: 한남대교 → 서울, 광안대교 → 부산)
반드시 아래 목록 중 하나만 답하고, 없으면 "없음"이라고만 답하세요.
목록: {', '.join(REGIONS)}

질문: {user_input}
답:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    region = response.text.strip()
    return region if region in REGIONS else None

def get_weibull_from_gp(inspection_years=None, condition_grades=None):
    if inspection_years is None:
        inspection_years = DEFAULT_INSPECTION["years"]
    if condition_grades is None:
        condition_grades = DEFAULT_INSPECTION["grades"]

    future_years_list = [31, 35, 40, 45, 50]
    future_years = np.array(future_years_list)
    mean, _ = gp_estimate(inspection_years, condition_grades, future_years)
    eta, beta = estimate_weibull_params(mean, future_years)
    return eta, beta

def analyze_bridge(region: str,
                   inspection_years=None,
                   condition_grades=None) -> str:
    # Step 1: 데이터 수집 (작품설명서 [2] LLM 데이터 수집 에이전트)
    aadt = get_aadt(region)
    n_ft = get_freeze_thaw(region)

    # Step 2: GP → 와이블 모수 역산 (작품설명서 5.4)
    eta, beta = get_weibull_from_gp(inspection_years, condition_grades)

    # Step 3: ETC 최적화 — GP 역산 모수 투입 (작품설명서 5.1~5.2)
    T_star = optimize_T(eta=eta, beta=beta, AADT=aadt, N_FT=n_ft)
    cost   = ETC(T_star, eta=eta, beta=beta, AADT=aadt, N_FT=n_ft)
    savings = calc_lcc_savings(T_star, eta=eta, beta=beta,
                               AADT=aadt, N_FT=n_ft)
    prob_20 = failure_prob(20, eta=eta, beta=beta, N_FT=n_ft)

    # Step 4: LLM 자연어 리포트
    prompt = f"""당신은 교량 정비 전문가입니다. 다음 분석 결과를 한국어로 쉽게 설명해주세요.

지역: {region}
일일 교통량(AADT): {aadt:,}대
연간 동결-해빙 횟수: {n_ft}회
역산된 와이블 모수: η={eta:.1f}년, β={beta:.2f}
최적 정비 주기: {T_star:.1f}년
연간 기대 비용: {cost:,.0f}원
현행 법정 주기(2년) 대비 30년 LCC 절감률: {savings:.1f}%
20년 후 고장확률: {prob_20:.1%}

이 결과가 의미하는 바를 3문장 이내로 설명해주세요."""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return response.text

def chat():
    print("교량 정비 AI 챗봇입니다. '종료'를 입력하면 끝납니다.")
    print("예시: '강원 교량 정비 시점 알려줘', '한남대교 정비 주기는?'")

    while True:
        user_input = input("\n질문: ").strip()
        if user_input == "종료":
            break
        if not user_input:
            continue

        # LLM이 지역명 추출 (단순 문자열 매칭 → LLM 위임)
        print("지역 분석 중...")
        region = extract_region_with_llm(user_input)

        if region:
            print(f"{region} 분석 중...")
            result = analyze_bridge(region)
            print(result)
        else:
            print("분석할 지역을 찾지 못했습니다. "
                  "시·도 이름을 포함해서 질문해주세요.\n"
                  "예: '강원도 교량 정비 시점 알려줘'")

if __name__ == "__main__":
    chat()