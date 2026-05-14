# 전체 파이프라인 실행 (main)
import numpy as np
from cost_function import optimize_T, ETC, failure_prob, calc_lcc_savings
from gp_estimator import gp_estimate, estimate_weibull_params
from data_loader import get_aadt, get_freeze_thaw

# 1. GP → 와이블 모수 역산 → ETC 파이프라인
print("=" * 55)
print("[1] GP 예측 → 와이블 모수 역산 → ETC 최적화 (서울)")
print("=" * 55)

inspection_years  = np.array([0, 5, 10, 15, 20, 25, 30])
condition_grades  = np.array([5.0, 4.8, 4.5, 4.1, 3.7, 3.3, 3.0])
future_years_list = [31, 35, 40, 45, 50]
future_years      = np.array(future_years_list)

# GP 예측
mean, std = gp_estimate(inspection_years, condition_grades, future_years)
print("\nGP 등급 예측:")
for yr, m, s in zip(future_years_list, mean, std):
    print(f"  {yr}년: {m:.2f} (±{s:.2f})")

# 와이블 모수 역산
eta, beta = estimate_weibull_params(mean, future_years)
print(f"\n역산된 와이블 모수: η={eta:.1f}년, β={beta:.2f}")

# GP 역산 모수를 ETC에 투입 → T* 산출
aadt_seoul = get_aadt("서울")
n_ft_seoul = get_freeze_thaw("서울")

T_star = optimize_T(eta=eta, beta=beta, AADT=aadt_seoul, N_FT=n_ft_seoul)
cost   = ETC(T_star, eta=eta, beta=beta, AADT=aadt_seoul, N_FT=n_ft_seoul)
savings = calc_lcc_savings(T_star, eta=eta, beta=beta,
                           AADT=aadt_seoul, N_FT=n_ft_seoul)

print(f"\n최적 정비 주기 (서울): {T_star:.1f}년")
print(f"최소 연간 기대비용: {cost:,.0f}원")
print(f"법정 주기 대비 LCC 절감률: {savings:.1f}%")

# 2. 지역별 최적 정비 주기 및 고장확률
print("\n" + "=" * 55)
print("[2] 지역별 최적 정비 주기 및 고장확률")
print("=" * 55)

regions_n_ft = {
    "서울": 30, "강원": 80, "제주": 10,
    "부산": 20, "대구": 25, "인천": 28,
    "광주": 15, "대전": 27, "울산": 22,
    "경기": 32, "충북": 35, "충남": 28,
    "전북": 18, "전남": 12, "경북": 40,
    "경남": 20, "세종": 27,
}

print(f"{'지역':<6} {'N_FT':>6} {'T*(년)':>8} {'20년후 고장확률':>16} {'LCC절감(%)':>12}")
print("-" * 55)
for region, n_ft in regions_n_ft.items():
    aadt = get_aadt(region)
    # 지역별로도 GP 역산 모수 사용 (동일 점검 이력 가정 — 실데이터 확보 시 교체)
    T = optimize_T(eta=eta, beta=beta, AADT=aadt, N_FT=n_ft)
    prob = failure_prob(20, eta=eta, beta=beta, N_FT=n_ft)
    sv   = calc_lcc_savings(T, eta=eta, beta=beta, AADT=aadt, N_FT=n_ft)
    print(f"{region:<6} {n_ft:>6} {T:>8.1f} {prob:>16.3f} {sv:>12.1f}")

# 3. API 데이터 확인 (fallback 포함)
print("\n" + "=" * 55)
print("[3] 데이터 로더 확인 (AADT / N_FT)")
print("=" * 55)
for region in regions_n_ft.keys():
    aadt  = get_aadt(region)
    n_ft  = get_freeze_thaw(region)
    print(f"  {region}: AADT={aadt:,}, N_FT={n_ft}")