#두 파일 불러서 실행하는 파일
import numpy as np
from cost_function import optimize_T, ETC, failure_prob
from gp_estimator import gp_estimate, estimate_weibull_params
from data_loader import get_aadt, get_freeze_thaw

# 최적 정비 시점 (정상 상황)
T_star = optimize_T()
print(f"최적 정비 주기 (서울): {T_star:.1f}년")
print(f"최소 연간 비용: {ETC(T_star):,.0f}원")

# GP 예측
inspection_years = np.array([[0], [5], [10], [15], [20]])
condition_grades = np.array([5.0, 4.5, 4.0, 3.2, 2.8])
future_years = np.array([[25], [30], [35]])

mean, std = gp_estimate(inspection_years, condition_grades, future_years)
for year, m, s in zip([25, 30, 35], mean, std):
    print(f"{year}년 예측 등급: {m:.2f} (±{s:.2f})")

eta, beta = estimate_weibull_params(mean, np.array([25, 30, 35]))
print(f"\n역산된 와이블 모수: η={eta:.1f}년, β={beta:.2f}")

# 지역별 최적 정비 주기 + 고장확률 비교
regions = {
    "서울": 30, "강원": 80, "제주": 10,
    "부산": 20, "대구": 25, "인천": 28,
    "광주": 15, "대전": 27, "울산": 22,
    "경기": 32, "충북": 35, "충남": 28,
    "전북": 18, "전남": 12, "경북": 40,
    "경남": 20, "세종": 27,
}

print("\n지역별 최적 정비 주기 및 고장확률")
for region, N_FT in regions.items():
    T = optimize_T(N_FT=N_FT)
    prob = failure_prob(20, N_FT=N_FT)
    print(f"{region} (N_FT={N_FT}): 정비주기 {T:.1f}년, 20년후 고장확률 {prob:.3f}")

# API 테스트 (키 없이 fallback 테스트)
print("\nAPI (전체 지역)")
for region in regions.keys():
    aadt = get_aadt(region)
    n_ft = get_freeze_thaw(region)
    print(f"{region}: AADT={aadt}, N_FT={n_ft}")