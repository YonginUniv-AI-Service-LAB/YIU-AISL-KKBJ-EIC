import numpy as np
import matplotlib.pyplot as plt
from baseline_policies_and_simulator import BaselineBridgeSimulator
from bridge_generator import generate_bridge_data
from cost_function import ETC
from gp_estimator import gp_estimate

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_simulation_and_plot_fig1():
    print("교량 데이터 생성 중...")
    bridges_df = generate_bridge_data(200)
    simulator = BaselineBridgeSimulator(bridges_df)

    policies = ["법정", "임계값", "Frangopol", "제안"]
    lccs = []
    fails = []
    
    for policy in policies:
        print(f"[{policy}] 정책 시뮬레이션 진행 중...")
        costs = []
        fail_counts = []
        # MC 100회
        for r in range(100):
            c, f = simulator.simulate_policy(policy, seed=r)
            costs.append(c)
            fail_counts.append(f)
            
        mean_lcc = np.mean(costs) / 1e8 # 억원 단위
        mean_fail = np.mean(fail_counts)
        lccs.append(mean_lcc)
        fails.append(mean_fail)
        print(f" -> LCC: {mean_lcc:.2f}억원, 고장횟수: {mean_fail:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 막대그래프: 평균 LCC
    bars = ax1.bar(policies, lccs, color=['#cccccc', '#ff9999', '#99ccff', '#ffcc99'], width=0.5)
    ax1.set_ylabel('30년 평균 LCC (억원)', fontsize=12)
    ax1.set_ylim(0, max(lccs) * 1.2 if max(lccs) > 0 else 100)
    
    # 꺾은선그래프: 고장횟수
    ax2 = ax1.twinx()
    lines = ax2.plot(policies, fails, color='red', marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('평균 고장횟수 (회)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(fails) * 1.2 if max(fails) > 0 else 1)
    
    plt.title('정책별 30년 생애주기비용(LCC) 및 고장횟수 비교', fontsize=15)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('figure_1_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("그림 1 저장 완료: figure_1_baseline_comparison.png")

def plot_fig2():
    print("ETC 곡선 및 GP 예측 그래프 생성 중...")
    T_vals = np.linspace(1, 20, 100)
    etc_vals = [ETC(t, AADT=50000, N_FT=30, alpha=0.002) for t in T_vals]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(T_vals, etc_vals, 'b-', lw=2)
    axes[0].set_title('통합 기대 비용(ETC) 곡선')
    axes[0].set_xlabel('정비 주기 (년)')
    axes[0].set_ylabel('연간 단위 비용 (원)')
    axes[0].grid(True, alpha=0.3)
    
    inspection_years = np.array([[0], [5], [10], [15], [20]])
    condition_grades = np.array([5.0, 4.5, 4.0, 3.2, 2.8])
    future_years = np.linspace(0, 35, 100).reshape(-1, 1)
    
    mean, std = gp_estimate(inspection_years, condition_grades, future_years)
    
    axes[1].plot(inspection_years, condition_grades, 'ro', label='실제 점검 데이터')
    axes[1].plot(future_years, mean, 'b-', label='GP 예측 평균')
    axes[1].fill_between(future_years.ravel(), mean - 1.96*std, mean + 1.96*std, color='blue', alpha=0.2, label='95% 신뢰구간')
    axes[1].axhline(2.0, color='r', linestyle='--', label='임계 등급 (D)')
    axes[1].set_title('GP 기반 교량 상태 등급 예측')
    axes[1].set_xlabel('경과 년수')
    axes[1].set_ylabel('상태 등급')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.savefig('figure_2_etc_gp.png', dpi=300, bbox_inches='tight')
    print("그림 2 저장 완료: figure_2_etc_gp.png")

if __name__ == "__main__":
    run_simulation_and_plot_fig1()
    plot_fig2()
