import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def perform_statistical_tests(results_dict):
    """
    통계적 유의성 검정
    - Friedman 검정: 4개 정책 동시 비교
    - Wilcoxon 사후검정 + Bonferroni 교정: 제안 vs 나머지 3개
    - 반환: (p_friedman, p_values_post)
      p_values_post 키: "임계값", "Frangopol", "제안"  ← 버그 수정
    """
    proposed = results_dict["제안"]

    # Friedman 검정 (4개 정책 동시 비교)
    stat, p_friedman = stats.friedmanchisquare(
        results_dict["법정"],
        results_dict["임계값"],
        results_dict["Frangopol"],
        results_dict["제안"],
    )

    # Wilcoxon 사후검정: 제안 vs 각 베이스라인
    # 본페로니 교정 n=3 (비교 쌍 수)
    p_values_post = {}
    comparisons = ["임계값", "Frangopol", "제안"]   # ← "법정" 제거, "제안" 추가
    n_comparisons = len(comparisons)

    for name in comparisons:
        if name == "제안":
            # 제안 vs 법정
            _, p_val = stats.wilcoxon(proposed, results_dict["법정"])
        else:
            _, p_val = stats.wilcoxon(proposed, results_dict[name])
        p_values_post[name] = min(1.0, p_val * n_comparisons)

    return p_friedman, p_values_post


def visualize_results(results_table, raw_results):
    """
    시각화:
    - 왼쪽: 법정 vs 제안 바이올린+스트립 분포 비교
    - 오른쪽: 4개 정책 평균 LCC 막대 그래프 + 고장횟수 보조축
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    UNIT = 1e8  # 억 원

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── 왼쪽: 분포 비교 (법정 vs 제안) ──────────────────────────────────────
    ax0 = axes[0]
    plot_data = pd.DataFrame(
        {k: v for k, v in raw_results.items() if k in ["법정", "제안"]}
    )
    melted = plot_data.melt(var_name="정책", value_name="LCC")
    melted["LCC_억"] = melted["LCC"] / UNIT

    sns.violinplot(
        x="정책", y="LCC_억", data=melted, ax=ax0,
        palette="Pastel1", inner="quartile", cut=0,
    )
    sns.stripplot(
        x="정책", y="LCC_억", data=melted, ax=ax0,
        color="black", size=3, alpha=0.3,
    )

    ax0.set_title("30년 누적 LCC 분포 비교\n(법정 vs 제안)", fontsize=13, fontweight='bold', pad=12)
    ax0.set_ylabel("총 LCC (억 원)", fontsize=11)
    ax0.set_xlabel("유지관리 정책", fontsize=11)

    # ── 오른쪽: 4개 정책 평균 LCC + 고장횟수 보조축 ─────────────────────────
    ax1 = axes[1]
    main_policies = ["법정", "임계값", "Frangopol", "제안"]
    means    = [results_table[p]["평균 LCC"] / UNIT for p in main_policies]
    failures = [results_table[p]["고장횟수"]        for p in main_policies]

    colors = ['#bdc3c7', '#bdc3c7', '#bdc3c7', '#e74c3c']
    x = np.arange(len(main_policies))
    bars = ax1.bar(x, means, color=colors, edgecolor="0.3", width=0.55, label="평균 LCC")

    # 막대 위 수치 레이블
    for bar, val in zip(bars, means):
        ax1.annotate(
            f'{val:.1f}억',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            xytext=(0, 4), textcoords='offset points',
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(main_policies, fontsize=11)
    ax1.set_ylabel("평균 비용 (억 원)", fontsize=11)
    ax1.set_xlabel("유지관리 정책", fontsize=11)
    ax1.set_title("정책별 평균 기대 LCC 비교", fontsize=13, fontweight='bold', pad=12)

    # 고장횟수 보조축 (꺾은선)
    ax1_r = ax1.twinx()
    ax1_r.plot(x, failures, color='#2980b9', marker='o', linewidth=2,
               markersize=6, label="기대 고장횟수")
    ax1_r.set_ylabel("30년 기대 고장횟수 (회)", fontsize=11, color='#2980b9')
    ax1_r.tick_params(axis='y', colors='#2980b9')

    # 범례 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # 전체 제목 (절감률)
    legal_lcc    = results_table["법정"]["평균 LCC"]
    proposed_lcc = results_table["제안"]["평균 LCC"]
    reduction    = (1 - proposed_lcc / legal_lcc) * 100
    fig.suptitle(
        f"시뮬레이션 결과 요약: 제안 정책 적용 시 법정 대비 약 {reduction:.1f}% 비용 절감",
        fontsize=14, fontweight='bold', y=1.02, color='#2c3e50',
    )

    plt.tight_layout()
    plt.savefig("figure_1_baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()