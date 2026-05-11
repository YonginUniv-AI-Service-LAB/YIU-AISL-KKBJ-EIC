import json
import numpy as np
import pandas as pd
import warnings
import unicodedata

# 기존 모듈 임포트
from cost_function import optimize_T, ETC, failure_prob
from gp_estimator import gp_estimate, estimate_weibull_params
from data_loader import get_aadt, get_freeze_thaw
from bridge_generator import generate_synthetic_bridges
from simulator import BridgeSimulator
from stats_analyzer import perform_statistical_tests, visualize_results

# 경고 메시지 숨김
warnings.filterwarnings("ignore")

def main():
    # --- [1] 분석 기초 데이터 및 예측 결과 출력 ---
    print("\n" + "="*95)
    print(" [1] 분석 기초 데이터 및 예측 결과")
    print("="*95)
    
    # 기초 연산 및 GP 예측
    T_star = optimize_T()
    print(f" > 최적 정비 주기 (서울): {T_star:.1f}년 | 최소 연간 비용: {ETC(T_star):,.0f}원")
    
    inspection_years = np.array([[0], [5], [10], [15], [20]])
    condition_grades = np.array([5.0, 4.5, 4.0, 3.2, 2.8])
    future_years = np.array([[25], [30], [35]])
    mean, std = gp_estimate(inspection_years, condition_grades, future_years)
    eta, beta = estimate_weibull_params(mean, np.array([25, 30, 35]))
    print(f" > 역산된 와이블 모수: η={eta:.1f}년, β={beta:.2f}")

    # --- [2] 시뮬레이션 수행 파트 ---
    bridges_df = generate_synthetic_bridges(200)
    simulator = BridgeSimulator(bridges_df)
    raw_results, results_table = {}, {}
    policies = ["법정", "임계값", "Frangopol", "제안"]

    for policy in policies:
        raw_costs, raw_fails = [], []
        for r in range(100):
            c, f = simulator.simulate_policy(policy, seed=r)
            raw_costs.append(c)
            raw_fails.append(f)
        raw_results[policy] = raw_costs
        results_table[policy] = {
            "평균 LCC": np.mean(raw_costs), "표준편차": np.std(raw_costs),
            "고장횟수": np.mean(raw_fails), "p-value": 1.0, "type": "main"
        }

    ablation_modes = {"-기상": "no_weather", "-기회비용": "no_opportunity", "-인력": "no_penalty"}
    for label, mode in ablation_modes.items():
        m_c, s_c, m_f = simulator.run_monte_carlo(policy_type="제안", ablation_mode=mode, iter_num=100)
        results_table[label] = {"평균 LCC": m_c, "표준편차": s_c, "고장횟수": m_f, "p-value": None, "type": "ablation"}

    _, p_post = perform_statistical_tests(raw_results)
    results_table["법정"]["p-value"] = "—"
    results_table["임계값"]["p-value"] = p_post["임계값"]
    results_table["Frangopol"]["p-value"] = p_post["Frangopol"]
    results_table["제안"]["p-value"] = p_post["법정"]

    # --- [3] 표 출력 파트: 정밀 수직 정렬 보정 ---
    UNIT = 1e8
    
    # 문자의 실제 출력 폭을 계산 (한글=2, 영문/기호/숫자=1)
    def get_width(text):
        width = 0
        for char in text:
            # ▶ 기호나 한글 등 동아시아 문자는 너비 2로 계산
            if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
                width += 2
            else:
                width += 1
        return width

    # 고정 폭 안에서 정렬을 수행하는 함수
    def pad(text, length, align='left'):
        w = get_width(text)
        if align == 'left':
            return text + ' ' * (length - w)
        else:
            return ' ' * (length - w) + text

    # 열 너비 설정
    W_POL = 22
    W_LCC = 22
    W_STD = 18
    W_FAL = 15
    W_PV  = 12

    print("\n" + "="*95)
    # 제목 줄 정렬
    header = (pad("관리 정책", W_POL) + " | " + 
              pad("평균 LCC (억 원)", W_LCC, 'right') + " | " + 
              pad("표준편차 (억)", W_STD, 'right') + " | " + 
              pad("고장횟수", W_FAL, 'right') + " | " + 
              pad("p-value", W_PV, 'right'))
    print(header)
    print("-" * 95)

    for k, v in results_table.items():
        if v.get("type") == "ablation" and k == "-기상":
            print("-" * 95)
            print(" [Ablation Study: 구성 요소 분석]")

        # 정책명 앞 특수기호 처리
        name_str = f"▶ {k}" if k == "제안" else f"  {k}"
        
        col_pol = pad(name_str, W_POL)
        col_lcc = pad(f"{v['평균 LCC'] / UNIT:,.1f}", W_LCC, 'right')
        col_std = pad(f"{v['표준편차'] / UNIT:,.2f}", W_STD, 'right')
        col_fal = pad(f"{v['고장횟수']:.4f}", W_FAL, 'right')
        
        # p-value 처리
        p_raw = v['p-value']
        if p_raw in ["—", None]:
            p_str = "—"
        else:
            p_val = float(p_raw)
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        col_pv = pad(p_str, W_PV, 'right')

        print(f"{col_pol} | {col_lcc} | {col_std} | {col_fal} | {col_pv}")

    print("="*95)
    print(" * 모든 비용은 30년 생애주기 기준 '억 원' 단위로 표시되었습니다.")
    print("="*95 + "\n")

    visualize_results(results_table, raw_results)

if __name__ == "__main__":
    main()