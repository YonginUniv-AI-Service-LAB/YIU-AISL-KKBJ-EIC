import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기존 모듈들 임포트
from cost_function import optimize_T, ETC, failure_prob
from gp_estimator import gp_estimate, estimate_weibull_params
from data_loader import get_aadt, get_freeze_thaw
from bridge_generator import generate_synthetic_bridges
from simulator import BridgeSimulator
from stats_analyzer import perform_statistical_tests, visualize_results

def main():
    print("\n" + "="*85)
    print(" [1] GP(가우스 과정) 예측 및 와이블 모수 분석")
    print("="*85)
    
    # GP 예측 수행
    inspection_years = np.array([[0], [5], [10], [15], [20]])
    condition_grades = np.array([5.0, 4.5, 4.0, 3.2, 2.8])
    future_years = np.array([[25], [30], [35]])

    mean, std = gp_estimate(inspection_years, condition_grades, future_years)
    for year, m, s in zip([25, 30, 35], mean, std):
        print(f" > {year}년 예측 등급: {m:.2f} (±{s:.2f})")

    eta, beta = estimate_weibull_params(mean, np.array([25, 30, 35]))
    print(f"\n 역산된 와이블 모수: η={eta:.1f}년, β={beta:.2f}")

    print("\n" + "="*85)
    print(" [2] 지역별 최적 정비 주기 (AADT 및 기상 데이터 반영)")
    print("="*85)
    regions = {
        "서울": 30, "강원": 80, "제주": 10, "부산": 20, "대구": 25, "인천": 28,
        "광주": 15, "대전": 27, "울산": 22, "경기": 32, "충북": 35, "충남": 28,
        "전북": 18, "전남": 12, "경북": 40, "경남": 20, "세종": 27,
    }
    print(f"{'지역':<6} | {'AADT':<10} | {'N_FT':<4} | {'최적 주기':<8} | {'고장확률(20y)':<10}")
    print("-" * 85)
    for region, n_ft in regions.items():
        aadt = get_aadt(region)
        t_opt = optimize_T(N_FT=n_ft)
        prob = failure_prob(20, N_FT=n_ft)
        print(f"{region:<6} | {aadt:>10,} | {n_ft:>4} | {t_opt:>6.1f}년 | {prob:>12.3f}")

    print("\n" + "="*85)
    print(" [3] 정책별 LCC 시뮬레이션 결과 (200개 교량 대상)")
    print("="*85)

    # 시뮬레이션 수행 데이터 준비
    bridges_df = generate_synthetic_bridges(200)
    simulator = BridgeSimulator(bridges_df)
    raw_results, results_table = {}, {}
    policies = ["법정", "임계값", "Frangopol", "제안"]

    # 기본 정책 시뮬레이션
    for policy in policies:
        raw_costs, raw_fails = [], []
        for r in range(100):
            c, f = simulator.simulate_policy(policy, seed=r)
            raw_costs.append(c)
            raw_fails.append(f)
        raw_results[policy] = raw_costs
        results_table[policy] = {
            "평균 LCC": np.mean(raw_costs),
            "표준편차": np.std(raw_costs),
            "고장횟수": np.mean(raw_fails),
            "p-value": 1.0, "type": "main"
        }

    # Ablation Study (구성 요소 분석)
    ablation_modes = {"-기상": "no_weather", "-기회비용": "no_opportunity", "-인력": "no_penalty"}
    for label, mode in ablation_modes.items():
        m_c, s_c, m_f = simulator.run_monte_carlo(policy_type="제안", ablation_mode=mode, iter_num=100)
        results_table[label] = {
            "평균 LCC": m_c, "표준편차": s_c, "고장횟수": m_f, "p-value": None, "type": "ablation"
        }

    # 통계 검정 및 p-value 업데이트
    p_friedman, p_post = perform_statistical_tests(raw_results)
    results_table["법정"]["p-value"] = "—"
    results_table["임계값"]["p-value"] = p_post["임계값"]
    results_table["Frangopol"]["p-value"] = p_post["Frangopol"]
    results_table["제안"]["p-value"] = p_post["법정"]

    # --- [표 출력 개선 부분] ---
    UNIT = 1e8  # 억 원 단위로 변환
    print(f"{'정책 이름':<15} | {'평균 LCC(억)':>15} | {'표준편차(억)':>10} | {'고장횟수':>10} | {'p-value':>10}")
    print("-" * 85)

    for name, data in results_table.items():
        # 요소 분석 섹션 구분선
        if data.get("type") == "ablation" and name == "-기상":
            print("-" * 85)
            print(" [구성 요소 분석: Ablation Study]")

        # 억 단위 변환 및 포맷팅
        lcc_won = f"{data['평균 LCC'] / UNIT:,.1f}"
        std_won = f"{data['표준편차'] / UNIT:,.2f}"
        fail_cnt = f"{data['고장횟수']:.4f}"
        
        # p-value 포맷팅
        if data['p-value'] in ["—", None]:
            p_str = "—"
        else:
            p_val = float(data['p-value'])
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

        # 제안 정책 강조
        display_name = f"▶ {name}" if name == "제안" else f"  {name}"
        print(f"{display_name:<15} | {lcc_won:>15} | {std_won:>10} | {fail_cnt:>10} | {p_str:>10}")

    print("="*85)
    print(" * LCC 단위: 억 원 (KRW 10^8)")
    print(" * p-value는 '제안' 정책 대비 통계적 유의성 검정 결과임")
    print("="*85)

    # 시각화 함수 호출
    visualize_results(results_table, raw_results)

if __name__ == "__main__":
    main()