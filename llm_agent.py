# LLM 챗봇 인터페이스
import os
import google.generativeai as genai
from dotenv import load_dotenv
from cost_function import optimize_T, ETC, failure_prob, calc_lcc_savings
from data_loader import get_aadt, get_freeze_thaw

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def analyze_bridge(region):
    aadt = get_aadt(region)
    n_ft = get_freeze_thaw(region)
    T_star = optimize_T(AADT=aadt, N_FT=n_ft)
    cost = ETC(T_star, AADT=aadt, N_FT=n_ft)
    savings = calc_lcc_savings(T_star, AADT=aadt, N_FT=n_ft)
    
    prompt = f"""
    당신은 교량 정비 전문가입니다. 다음 분석 결과를 한국어로 쉽게 설명해주세요.
    
    지역: {region}
    일일 교통량(AADT): {aadt:,}대
    연간 동결-해빙 횟수: {n_ft}회
    최적 정비 주기: {T_star:.1f}년
    연간 기대 비용: {cost:,.0f}원
    현행 법정 주기 대비 30년 LCC 절감률: {savings:.1f}%
    
    이 결과가 의미하는 바를 2-3문장으로 설명해주세요.
    """
    
    response = model.generate_content(prompt)
    return response.text

def chat():
    print("교량 정비 AI 챗봇입니다. '종료'를 입력하면 끝납니다.")
    while True:
        user_input = input("\n질문: ")
        if user_input == "종료":
            break
        
        regions = ["서울", "강원", "제주", "부산", "대구", "인천", 
                  "광주", "대전", "울산", "경기", "충북", "충남",
                  "전북", "전남", "경북", "경남", "세종"]
        
        found_region = None
        for r in regions:
            if r in user_input:
                found_region = r
                break
        
        if found_region:
            print(f"\n{found_region} 분석 중...")
            result = analyze_bridge(found_region)
            print(result)
        else:
            print("지역명을 포함해서 질문해주세요! 예: '서울 교량 정비 시점 알려줘'")

if __name__ == "__main__":
    chat()