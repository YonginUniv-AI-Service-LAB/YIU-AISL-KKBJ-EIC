import numpy as np
import matplotlib.pyplot as plt
from baseline_policies_and_simulator import BaselineBridgeSimulator, policy_threshold
from bridge_generator import generate_bridge_data
from cost_function import ETC
from gp_estimator import gp_estimate

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def grid_search_threshold(bridges_df, grade_candidates=None, n_mc=30):
    """
    임계값 grid search: 등급 후보군을 순회하며 30년 MC 평균 LCC가 가장 낮은 임계 등급을 반환.
    n_mc: grid search 속도를 위해 MC 횟수를 축소 (기본 30회)
    """
    if grade_candidates is None:
        grade_candidates = np.arange(1.5, 4.1, 0.1)  # 1.5 ~ 4.0 (0.1 단위)

    best_grade = 2.0
    best_lcc = float('inf')

    simulator = BaselineBridgeSimulator(bridges_df)
    print("  [임계값 grid search 진행 중...]")
    for grade in grade_candidates:
        eta = bridges_df['base_eta'].values if 'base_eta' in bridges_df.columns else bridges_df['eta'].values
        beta = bridges_df['beta'].values
        n_ft = bridges_df['n_ft'].values if 'n_ft' in bridges_df.columns else bridges_df['freeze_thaw_cycles'].values

        # 임계값별 고정 주기를 simulator에 넘기기 위해 직접 계산해 policy_threshold 결과로 끼워넣음
        costs = []
        for r in range(n_mc):
            np.random.seed(r)
            # simulate_policy는 policy_type="임계값"일 때 threshold_grade=2.0으로 고정되어 있으므로
            # 이 grid search에서는 직접 시뮬레이션 루프를 모방해 grade별로 비교
            from cost_function import optimize_T
            n_bridges = len(bridges_df)
            total_cost = 0.0
            u = np.random.uniform(0.01, 0.99, n_bridges)
            ttf = eta * (-np.log(1.0 - u)) ** (1.0 / beta)
            cm_val = 50000000; cf_val = 50000000000; alpha_val = 0.002; years = 30

            for i in range(n_bridges):
                t_star = policy_threshold(eta[i], beta[i], threshold_grade=grade, N_FT=n_ft[i], alpha=alpha_val)
                t_curr = 0; fail_time = ttf[i]
                while t_curr < years:
                    next_m = t_curr + t_star
                    if fail_time < next_m and fail_time < years:
                        total_cost += ETC(T=t_star, AADT=50000, Cf=cf_val, N_FT=n_ft[i], alpha=alpha_val)
                        t_curr = fail_time
                        fail_time = t_curr + eta[i] * (-np.log(1.0 - np.random.uniform(0.1, 0.99))) ** (1.0 / beta[i])
                    elif next_m < years:
                        total_cost += cm_val / t_star
                        t_curr = next_m
                    else:
                        break
            costs.append(total_cost)

        mean_lcc = np.mean(costs)
        if mean_lcc < best_lcc:
            best_lcc = mean_lcc
            best_grade = grade

    print(f"  -> 최적 임계 등급: {best_grade:.1f} (LCC: {best_lcc/1e8:.2f}억원)")
    return best_grade

def run_simulation_and_plot_fig1():
    print("교량 데이터 생성 중...")
    bridges_df = generate_bridge_data(200)
    simulator = BaselineBridgeSimulator(bridges_df)

    # 1) 임계값 grid search로 최적 임계 등급 탐색
    best_threshold_grade = grid_search_threshold(bridges_df)

    policies = ["법정", "임계값", "Frangopol", "제안"]
    lccs = []
    fails = []
    raw_costs_dict = {}  # Pareto frontier용

    for policy in policies:
        print(f"[{policy}] 정책 시뮬레이션 진행 중...")
        costs = []
        fail_counts = []
        for r in range(100):
            if policy == "임계값":
                # 최적 임계값으로 직접 시뮬레이션
                np.random.seed(r)
                eta = bridges_df['base_eta'].values if 'base_eta' in bridges_df.columns else bridges_df['eta'].values
                beta = bridges_df['beta'].values
                n_ft = bridges_df['n_ft'].values if 'n_ft' in bridges_df.columns else bridges_df['freeze_thaw_cycles'].values
                n_bridges = len(bridges_df)
                total_cost = 0.0; fail_cnt = 0.0
                cm_val = 50000000; cf_val = 50000000000; alpha_val = 0.002; years = 30
                u = np.random.uniform(0.01, 0.99, n_bridges)
                ttf = eta * (-np.log(1.0 - u)) ** (1.0 / beta)
                for i in range(n_bridges):
                    t_star = policy_threshold(eta[i], beta[i], threshold_grade=best_threshold_grade, N_FT=n_ft[i], alpha=alpha_val)
                    t_curr = 0; fail_time = ttf[i]
                    while t_curr < years:
                        next_m = t_curr + t_star
                        if fail_time < next_m and fail_time < years:
                            fail_cnt += 1
                            total_cost += ETC(T=t_star, AADT=50000, Cf=cf_val, N_FT=n_ft[i], alpha=alpha_val)
                            t_curr = fail_time
                            fail_time = t_curr + eta[i] * (-np.log(1.0 - np.random.uniform(0.1, 0.99))) ** (1.0 / beta[i])
                        elif next_m < years:
                            total_cost += cm_val / t_star
                            t_curr = next_m
                        else:
                            break
                costs.append(total_cost)
                fail_counts.append(fail_cnt / n_bridges)
            else:
                c, f = simulator.simulate_policy(policy, seed=r)
                costs.append(c)
                fail_counts.append(f)

        mean_lcc = np.mean(costs) / 1e8
        mean_fail = np.mean(fail_counts)
        lccs.append(mean_lcc)
        fails.append(mean_fail)
        raw_costs_dict[policy] = (mean_lcc, mean_fail)
        print(f" -> LCC: {mean_lcc:.2f}억원, 고장횟수: {mean_fail:.4f}")

    # ---- 그림 1-A: 막대그래프 + Pareto frontier (2x1 subplot) ----
    fig, (ax_bar, ax_pareto) = plt.subplots(1, 2, figsize=(16, 6))

    # 막대그래프
    colors = ['#cccccc', '#ff9999', '#99ccff', '#ffcc99']
    bars = ax_bar.bar(policies, lccs, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax_bar.set_ylabel('30년 평균 LCC (억원)', fontsize=12)
    ax_bar.set_ylim(0, max(lccs) * 1.25)
    ax_bar.set_title('정책별 30년 생애주기비용(LCC) 비교', fontsize=13)
    ax_bar.grid(axis='y', alpha=0.3)

    # 막대 위에 절감률 표시 (법정 대비)
    legal_lcc = lccs[0]
    for j, (bar, lcc) in enumerate(zip(bars, lccs)):
        pct = (legal_lcc - lcc) / legal_lcc * 100
        label = f'{lcc:.0f}억\n({pct:+.1f}%)' if j > 0 else f'{lcc:.0f}억'
        ax_bar.text(bar.get_x() + bar.get_width()/2, lcc + max(lccs)*0.02,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 고장횟수 꺾은선 (우축)
    ax_bar2 = ax_bar.twinx()
    ax_bar2.plot(policies, fails, color='red', marker='o', linewidth=2, markersize=8, label='평균 고장횟수')
    ax_bar2.set_ylabel('평균 고장횟수 (회)', fontsize=12, color='red')
    ax_bar2.tick_params(axis='y', labelcolor='red')
    ax_bar2.set_ylim(0, max(fails) * 1.5 if max(fails) > 0 else 1)

    # Pareto frontier
    ax_pareto.set_title('Pareto Frontier (LCC vs 고장횟수)', fontsize=13)
    for j, (policy, color) in enumerate(zip(policies, colors)):
        lcc_v, fail_v = raw_costs_dict[policy]
        ax_pareto.scatter(fail_v, lcc_v, color=color, s=150, edgecolors='black', linewidth=1.2, zorder=5)
        ax_pareto.annotate(policy, (fail_v, lcc_v),
                           textcoords="offset points", xytext=(8, 4), fontsize=11)

    # Pareto 선 (LCC 오름차순 정렬 후 연결)
    sorted_pts = sorted(raw_costs_dict.values(), key=lambda x: x[1])
    pareto_xs = [p[1] for p in sorted_pts]
    pareto_ys = [p[0] for p in sorted_pts]
    ax_pareto.plot(pareto_xs, pareto_ys, 'k--', linewidth=1, alpha=0.5)
    ax_pareto.set_xlabel('평균 고장횟수 (회)', fontsize=12)
    ax_pareto.set_ylabel('30년 평균 LCC (억원)', fontsize=12)
    ax_pareto.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure_1_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("그림 1 저장 완료: figure_1_baseline_comparison.png")
    print(f"  [결과 요약] 법정 대비 제안 절감률: {(lccs[0]-lccs[3])/lccs[0]*100:.1f}%")
    print(f"  [결과 요약] Frangopol 대비 제안 절감률: {(lccs[2]-lccs[3])/lccs[2]*100:.1f}%")

def plot_fig2():
    print("ETC 곡선 및 GP 예측 그래프 생성 중...")
    T_vals = np.linspace(1, 20, 100)

    # 비용 컴포넌트별 분해
    etc_total = [ETC(t, AADT=50000, N_FT=30, alpha=0.002) for t in T_vals]
    etc_maint  = [50000000 / t for t in T_vals]
    etc_fail   = [ETC(t, AADT=50000, N_FT=30, alpha=0.002, VOT=0, lam=0) - 50000000/t for t in T_vals]
    etc_user   = [ETC(t, AADT=50000, N_FT=30, alpha=0.002) - ETC(t, AADT=50000, N_FT=30, alpha=0.002, VOT=0, lam=0) for t in T_vals]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ETC 컴포넌트 분해 곡선
    axes[0].plot(T_vals, etc_total, 'k-', lw=2.5, label='총 ETC')
    axes[0].plot(T_vals, etc_maint, 'b--', lw=1.5, label='정비비 (Cm/T)')
    axes[0].plot(T_vals, etc_fail, 'r--', lw=1.5, label='고장비 (Frangopol)')
    axes[0].plot(T_vals, etc_user, 'g--', lw=1.5, label='사용자비용+페널티')
    axes[0].set_title('통합 기대 비용(ETC) 곡선 — 컴포넌트 분해', fontsize=12)
    axes[0].set_xlabel('정비 주기 T (년)', fontsize=11)
    axes[0].set_ylabel('연간 단위 비용 (원)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # GP 점검 등급 예측
    inspection_years = np.array([[0], [5], [10], [15], [20]])
    condition_grades = np.array([5.0, 4.5, 4.0, 3.2, 2.8])
    future_years = np.linspace(0, 35, 100).reshape(-1, 1)
    mean, std = gp_estimate(inspection_years, condition_grades, future_years)

    axes[1].plot(inspection_years, condition_grades, 'ro', markersize=8, label='실제 점검 데이터', zorder=5)
    axes[1].plot(future_years, mean, 'b-', lw=2, label='GP 예측 평균')
    axes[1].fill_between(future_years.ravel(), mean - 1.96*std, mean + 1.96*std,
                         color='blue', alpha=0.15, label='95% 신뢰구간')
    axes[1].axhline(2.0, color='r', linestyle='--', lw=1.5, label='임계 등급 D (=2.0)')
    axes[1].set_title('GP 기반 교량 상태 등급 예측', fontsize=12)
    axes[1].set_xlabel('경과 년수', fontsize=11)
    axes[1].set_ylabel('상태 등급 (A=5, E=1)', fontsize=11)
    axes[1].set_ylim(0.5, 5.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure_2_etc_gp.png', dpi=300, bbox_inches='tight')
    print("그림 2 저장 완료: figure_2_etc_gp.png")

if __name__ == "__main__":
    run_simulation_and_plot_fig1()
    plot_fig2()
