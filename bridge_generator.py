import numpy as np
import pandas as pd
from data_loader import DUMMY_AADT, DUMMY_N_FT

# ── 안전등급별 초기 분포 (가정값, 실데이터 확보 시 교체 필요) ─────────────────
GRADE_PROBS = {
    'A': 0.15,
    'B': 0.40,
    'C': 0.30,
    'D': 0.12,
    'E': 0.03,
}

# 등급별 η 보정 계수: 낮은 등급일수록 특성수명 단축
GRADE_ETA_FACTOR = {
    'A': 1.20,
    'B': 1.05,
    'C': 1.00,
    'D': 0.85,
    'E': 0.70,
}

# 합성 점검 이력 (GP 역산에 사용, 작품설명서 6.5)
# 준공~30년까지 5년 간격 7포인트, 5.0→3.0 완만 감소
_INSPECTION_YEARS = np.array([[0], [5], [10], [15], [20], [25], [30]])
_CONDITION_GRADES = np.array([5.0, 4.7, 4.4, 4.0, 3.6, 3.3, 3.0])
_FUTURE_YEARS     = np.array([[35], [40], [45]])


def generate_synthetic_bridges(n=200, seed=42, use_gp_params=False):
    """
    합성 교량 n개 생성

    Parameters
    ----------
    n             : 교량 수 (기본 200)
    seed          : 난수 시드
    use_gp_params : True면 GP 역산 분포에서 (η, β) 샘플링 (수정사항 문서 [5])
                    False면 문헌 기반 균등분포 샘플링 (기존 방식, 기본값)

    Columns
    -------
    bridge_id, region, AADT, freeze_thaw_cycles, eta, beta, grade
    """
    np.random.seed(seed)

    grades      = list(GRADE_PROBS.keys())
    grade_probs = list(GRADE_PROBS.values())
    regions     = list(DUMMY_AADT.keys())

    # ── GP 역산 분포 사전 준비 ─────────────────────────────────────────────
    use_gp = False
    if use_gp_params:
        try:
            from gp_estimator import estimate_weibull_distribution
            gp_etas, gp_betas = estimate_weibull_distribution(
                _INSPECTION_YEARS, _CONDITION_GRADES, _FUTURE_YEARS,
                n_samples=max(n * 3, 600), seed=seed,
            )
            if len(gp_etas) >= 10:
                use_gp = True
                print(f"[bridge_generator] GP 역산 성공: {len(gp_etas)}개 유효 샘플 사용")
            else:
                print("[bridge_generator] GP 샘플 부족 → 문헌 기반 균등분포 사용")
        except Exception as e:
            print(f"[bridge_generator] GP 역산 실패 → 문헌 기반 균등분포 사용: {e}")

    bridges = []
    for i in range(n):
        region = np.random.choice(regions)

        weather_cycle = max(1, int(np.random.normal(DUMMY_N_FT[region], 5)))
        aadt = max(1000, int(np.random.normal(
            DUMMY_AADT[region], DUMMY_AADT[region] * 0.15
        )))
        grade = np.random.choice(grades, p=grade_probs)

        if use_gp:
            # GP 역산 분포에서 무작위 선택 후 등급 보정
            idx  = np.random.randint(len(gp_etas))
            eta  = gp_etas[idx] * GRADE_ETA_FACTOR[grade]
            beta = float(np.clip(gp_betas[idx], 1.5, 10.0))
        else:
            # 문헌 기반 균등분포 (Grussing & Marrano 2006)
            eta_base = np.random.uniform(25.0, 45.0)
            eta      = eta_base * GRADE_ETA_FACTOR[grade]
            beta     = np.random.uniform(2.0, 3.5)

        bridges.append({
            "bridge_id":          i,
            "region":             region,
            "AADT":               aadt,
            "freeze_thaw_cycles": weather_cycle,
            "eta":                eta,
            "beta":               beta,
            "grade":              grade,
        })

    return pd.DataFrame(bridges)


if __name__ == "__main__":
    print("=== 기본 모드 (문헌 기반 균등분포) ===")
    df = generate_synthetic_bridges(200, use_gp_params=False)
    print(df.head(5).to_string())
    print("\n등급 분포:")
    print(df['grade'].value_counts().sort_index())
    print(f"\nη 범위: {df['eta'].min():.1f} ~ {df['eta'].max():.1f}년")

    print("\n=== GP 역산 모드 ===")
    df_gp = generate_synthetic_bridges(200, use_gp_params=True)
    print(f"η 범위: {df_gp['eta'].min():.1f} ~ {df_gp['eta'].max():.1f}년")
    print(f"β 범위: {df_gp['beta'].min():.2f} ~ {df_gp['beta'].max():.2f}")