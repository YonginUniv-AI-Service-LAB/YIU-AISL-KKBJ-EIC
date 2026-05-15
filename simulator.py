import numpy as np
import cost_function

# ── 등급별 법정 점검 주기 (시설물안전법 시행령 별표 3, 교량 기준) ──────────────
# 교량의 안전등급별 정밀안전점검 주기
LEGAL_CYCLE = {
    'A': 3.0,   # A등급: 3년
    'B': 2.0,   # B등급: 2년
    'C': 2.0,   # C등급: 2년
    'D': 1.0,   # D등급: 1년
    'E': 1.0,   # E등급: 1년
}

# 임계값 정책: 등급 D(점수 2.0) 이하 도달 시 정비
# 등급 점수 5=A, 4=B, 3=C, 2=D, 1=E 기준
# 등급 D 도달 = 고장확률 약 0.75로 근사
THRESHOLD_PROB = 0.75


class BridgeSimulator:
    def __init__(self, bridges_df):
        self.bridges = bridges_df
        self.n_bridges = len(bridges_df)

    def _get_t_star(self, b, policy_type, cm_val, cf_val, vot_val, alpha_val, lam_val):
        """정책별 최적 정비 주기 T* 산출"""

        if policy_type == "법정":
            # 등급별 차등 주기 적용 (컬럼 없으면 B·C 평균 2년 기본값)
            grade = b.get('grade', 'C') if hasattr(b, 'get') else getattr(b, 'grade', 'C')
            return LEGAL_CYCLE.get(grade, 2.0)

        elif policy_type == "임계값":
            # 와이블 누적분포로 등급 D(고장확률 THRESHOLD_PROB) 도달 시점 계산
            # F(t) = 1 - exp(-(t/eta)^beta) = 0.75
            # → t = eta * (-ln(1 - 0.75))^(1/beta)
            eta_local = b['eta'] * np.exp(-alpha_val * b['freeze_thaw_cycles'])
            t_threshold = eta_local * (-np.log(1 - THRESHOLD_PROB)) ** (1.0 / b['beta'])
            # 탐색 범위 [1, 60] 클리핑
            return float(np.clip(t_threshold, 1.0, 60.0))

        elif policy_type == "Frangopol":
            # 작품설명서 6.3: 사용자 기회비용(VOT=0)·인력 페널티(lam=0) 제거 후 재최적화
            # → Frangopol et al.(1997) 원형인 ①+② only 구조
            return cost_function.optimize_T(
                AADT=b['AADT'],
                C_m=cm_val,
                C_f=cf_val,
                VOT=0,          # 사용자 기회비용 비활성화
                N_FT=b['freeze_thaw_cycles'],
                alpha=alpha_val,
                lam=0,          # 인력 페널티 비활성화
            )

        else:  # "제안" (Full 모델)
            return cost_function.optimize_T(
                AADT=b['AADT'],
                C_m=cm_val,
                C_f=cf_val,
                VOT=vot_val,
                N_FT=b['freeze_thaw_cycles'],
                alpha=alpha_val,
                lam=lam_val,
            )

    def simulate_policy(self, policy_type="제안", ablation_mode=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        years = 30

        total_costs   = np.zeros(self.n_bridges)
        failures_count = np.zeros(self.n_bridges)

        # ── 기본 파라미터 ──────────────────────────────────────────────────────
        cm_val    = 300_000_000      # 정비비 3억원 (한국건설기술연구원 표준품셈 참조)
        cf_val    = None             # cost_function.calc_cf()로 AADT별 자동 계산
        vot_val   = 11_000           # 통행시간가치 (원/시간, KDI 표 Ⅵ-6)
        alpha_val = 0.002            # 동결-해빙 보정계수 (가정값, 민감도 분석 대상)
        lam_val   = 1_000_000        # 인력 페널티 승수 (가정값, 민감도 분석 대상)

        # ── Ablation: 해당 항 비활성화 ─────────────────────────────────────────
        if ablation_mode == "no_weather":
            alpha_val = 0
        elif ablation_mode == "no_opportunity":
            vot_val = 0
        elif ablation_mode == "no_penalty":
            lam_val = 0

        for i in range(self.n_bridges):
            b = self.bridges.iloc[i]

            # ── 정책별 T* 산출 ─────────────────────────────────────────────────
            t_star = self._get_t_star(
                b, policy_type, cm_val, cf_val, vot_val, alpha_val, lam_val
            )

            # ── 연간 기대 비용 (결정론적) ──────────────────────────────────────
            annual_cost = cost_function.ETC(
                T=t_star,
                AADT=b['AADT'],
                C_m=cm_val,
                C_f=cf_val,
                VOT=vot_val,
                N_FT=b['freeze_thaw_cycles'],
                alpha=alpha_val,
                lam=lam_val,
            )

            # ── [핵심 수정] 와이블 분포로 실제 고장 시점 샘플링 ────────────────
            # 역변환 샘플링: u ~ Uniform(0,1), t_fail = η_local * (-ln(u))^(1/β)
            eta_local = b['eta'] * np.exp(-alpha_val * b['freeze_thaw_cycles'])
            u = np.random.uniform()
            # u=0이면 log(0) → -inf 방지
            u = np.clip(u, 1e-9, 1 - 1e-9)
            failure_time = eta_local * (-np.log(u)) ** (1.0 / b['beta'])

            # 30년 내 고장 발생 시 긴급 보수비 추가
            if failure_time < years:
                emergency_cost = cost_function.calc_cf(b['AADT'], VOT=vot_val)
                failures_count[i] = 1
            else:
                emergency_cost = 0
                failures_count[i] = 0

            total_costs[i] = annual_cost * years + emergency_cost

        return total_costs.mean(), failures_count.mean()

    def run_monte_carlo(self, policy_type="제안", ablation_mode=None, iter_num=100):
        costs, failures = [], []
        for run in range(iter_num):
            cost, fail = self.simulate_policy(policy_type, ablation_mode, seed=run)
            costs.append(cost)
            failures.append(fail)
        return np.mean(costs), np.std(costs), np.mean(failures)
