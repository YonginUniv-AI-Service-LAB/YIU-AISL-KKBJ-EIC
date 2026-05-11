import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def perform_statistical_tests(results_dict):
    """통계적 유의성 검정 (기존 로직 유지)"""
    legal, proposed = results_dict["법정"], results_dict["제안"]
    stat, p_friedman = stats.friedmanchisquare(
        results_dict["법정"], results_dict["임계값"], 
        results_dict["Frangopol"], results_dict["제안"]
    )
    p_values_post = {}
    for name in ["법정", "임계값", "Frangopol"]:
        _, p_val = stats.wilcoxon(proposed, results_dict[name])
        # 본페로니 교정 (Bonferroni correction)
        p_values_post[name] = min(1.0, p_val * 3)
    return p_friedman, p_values_post

def visualize_results(results_table, raw_results):
    """
    디벨롭된 시각화 함수:
    1. Y축 단위를 '억 원'으로 변환하여 가독성 증대
    2. 바이올린 플롯과 개별 포인트를 결합하여 분포 시각화
    3. 막대 그래프에 수치 레이블 추가
    """
    # 한글 폰트 설정 및 스타일 적용
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 단위 변환: 원 -> 억 원 (10^8)
    UNIT = 1e8
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- [왼쪽 그래프] 분포 비교 (법정 vs 제안) ---
    # 데이터 프레임 구성
    plot_data = pd.DataFrame({k: v for k, v in raw_results.items() if k in ["법정", "제안"]})
    plot_data_melted = plot_data.melt(var_name="정책", value_name="LCC")
    plot_data_melted["LCC_KRW"] = plot_data_melted["LCC"] / UNIT

    # 바이올린 플롯 + 박스플롯 결합
    sns.violinplot(x="정책", y="LCC_KRW", data=plot_data_melted, ax=ax[0], 
                   palette="Pastel1", inner="quartile", cut=0)
    sns.stripplot(x="정책", y="LCC_KRW", data=plot_data_melted, ax=ax[0], 
                  color="black", size=3, alpha=0.3) # 개별 데이터 포인트 표시
    
    ax[0].set_title("30년 누적 LCC 분포 상세 비교", fontsize=14, fontweight='bold', pad=15)
    ax[0].set_ylabel("총 LCC (억 원)", fontsize=12)
    ax[0].set_xlabel("유지관리 정책", fontsize=12)

    # --- [오른쪽 그래프] 전체 정책 평균 비교 ---
    policies = list(results_table.keys())[:4]
    means = [results_table[p]["평균 LCC"] / UNIT for p in policies]
    
    # 색상 강조 (제안 정책만 다른 색상)
    colors = ['#bdc3c7', '#bdc3c7', '#bdc3c7', '#e74c3c'] 
    
    bars = sns.barplot(x=policies, y=means, palette=colors, ax=ax[1], edgecolor="0.3")
    
    # 막대 위에 수치 레이블 표시
    for bar in bars.patches:
        ax[1].annotate(f'{bar.get_height():.1f}억', 
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='center', size=11, xytext=(0, 8),
                       textcoords='offset points', fontweight='bold')

    ax[1].set_title("정책별 평균 기대 LCC 비교", fontsize=14, fontweight='bold', pad=15)
    ax[1].set_ylabel("평균 비용 (억 원)", fontsize=12)
    ax[1].set_xlabel("유지관리 정책", fontsize=12)
    
    # 절감률 텍스트 추가 (법정 대비 제안)
    reduction = (1 - (results_table["제안"]["평균 LCC"] / results_table["법정"]["평균 LCC"])) * 100
    plt.suptitle(f"시뮬레이션 결과 요약: 제안 정책 적용 시 법정 대비 약 {reduction:.1f}% 비용 절감", 
                 fontsize=16, fontweight='bold', y=1.05, color='#2c3e50')

    plt.tight_layout()
    plt.savefig("simulation_result_updated.png", dpi=300, bbox_inches='tight')
    plt.show()