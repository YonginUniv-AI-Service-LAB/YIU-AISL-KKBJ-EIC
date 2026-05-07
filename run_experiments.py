import json
import numpy as np
from bridge_generator import generate_synthetic_bridges
from simulator import BridgeSimulator
from stats_analyzer import perform_statistical_tests

def main():
    print("=================== 인프라 LCC 시뮬레이션 실험 시작 ===================")
    
    bridges_df = generate_synthetic_bridges(200)
    simulator = BridgeSimulator(bridges_df)
    
    raw_results = {}
    results_table = {}
    
    policies = ["법정", "임계값", "Frangopol", "제안"]
    
    for policy in policies:
        print(f"[{policy}] 정책 시뮬레이션 진행 중...")
        raw_costs = []
        raw_fails = []
        for r in range(100):
            c, f = simulator.simulate_policy(policy, seed=r)
            raw_costs.append(c)
            raw_fails.append(f)
            
        raw_results[policy] = raw_costs
        results_table[policy] = {
            "평균 LCC": float(np.mean(raw_costs)),
            "표준편차": float(np.std(raw_costs)),
            "고장횟수": float(np.mean(raw_fails)),
            "p-value": 1.0
        }

    ablation_modes = {
        "-기상": "no_weather",
        "-기회비용": "no_opportunity",
        "-인력": "no_penalty"
    }
    
    for label, mode in ablation_modes.items():
        print(f"[Ablation: {label}] 시뮬레이션 진행 중...")
        mean_c, std_c, mean_f = simulator.run_monte_carlo(policy_type="제안", ablation_mode=mode, iter_num=100)
        results_table[label] = {
            "평균 LCC": float(mean_c),
            "표준편차": float(std_c),
            "고장횟수": float(mean_f),
            "p-value": None
        }

    print("\n[통계 분석] Friedman 및 Wilcoxon 사후 검정 진행 중...")
    p_friedman, p_post = perform_statistical_tests(raw_results)
    
    results_table["법정"]["p-value"] = "—"
    results_table["임계값"]["p-value"] = float(p_post["임계값"])
    results_table["Frangopol"]["p-value"] = float(p_post["Frangopol"])
    results_table["제안"]["p-value"] = float(p_post["법정"])

    print("\n" + "="*70)
    print(f"{'정책':<15} | {'평균 LCC':<10} | {'표준편차':<8} | {'고장횟수':<8} | {'p-value':<10}")
    print("-"*70)
    for k, v in results_table.items():
        p_val = v['p-value']
        p_str = "—" if p_val == "—" or p_val is None else f"{p_val:.4f}"
        print(f"{k:<15} | {v['평균 LCC']:<10.2f} | {v['표준편차']:<8.2f} | {v['고장횟수']:<8.4f} | {p_str:<10}")
    print("="*70)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=4, ensure_ascii=False)
    print("\n✔ 모든 시뮬레이션 결과가 'results.json'에 정상 저장되었습니다!")

if __name__ == "__main__":
    main()