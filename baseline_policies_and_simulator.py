import numpy as np
import cost_function
from simulator import BridgeSimulator

def policy_legal(bridge_eta, bridge_beta, N_FT, alpha):
    """
    현행 법정 정책: 시설물안전법 시행령 별표 3에 따라 등급별 고정 주기로 정비 시행.
    교량은 A=3년, B·C=2년, D·E=1년.
    단순화를 위해, 각 교량의 평균 생애 등급(C등급 수준)을 가정하여 2년을 기본으로 적용.
    """
    return 2.0

def policy_threshold(eta, beta, threshold_grade=2.0, N_FT=30, alpha=0.002):
    """
    단순 임계값 휴리스틱: 등급 D(2.0) 이하 도달 시 정비.
    목표 등급 도달 시간을 계산하여 해당 주기로 정비.
    """
    target_prob = (5.0 - threshold_grade) / 4.0
    eta_local = eta * np.exp(-alpha * N_FT)
    T = eta_local * (-np.log(1 - target_prob)) ** (1.0 / beta)
    return T

def policy_frangopol(AADT, Cm, Cf, N_FT, alpha):
    """
    Frangopol 1997 와이블 LCC: 
    통합 비용 함수에서 사용자 기회비용(VOT=0)·인력 페널티(lam=0) 항을 제거한 형태.
    """
    return cost_function.optimize_T(
        AADT=AADT, Cm=Cm, Cf=Cf, VOT=0, w=1.0, W_max=2.0, lam=0, N_FT=N_FT, alpha=alpha
    )

class BaselineBridgeSimulator(BridgeSimulator):
    """
    기존 BridgeSimulator를 상속받아 기존 코드를 수정하지 않고
    새로운 정책 함수들만 끼워넣어 시뮬레이션을 수행하는 래퍼 클래스.
    """
    def __init__(self, bridges_df):
        df = bridges_df.copy()
        # 원본 데이터 제너레이터와 원본 시뮬레이터 간의 컬럼명 불일치를 외부에서 해결
        if 'base_eta' in df.columns and 'eta' not in df.columns:
            df['eta'] = df['base_eta']
        if 'aadt' in df.columns and 'AADT' not in df.columns:
            df['AADT'] = df['aadt']
        if 'n_ft' in df.columns and 'freeze_thaw_cycles' not in df.columns:
            df['freeze_thaw_cycles'] = df['n_ft']
            
        super().__init__(df)

    def simulate_policy(self, policy_type="제안", ablation_mode=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        years = 30
        total_costs = np.zeros(self.n_bridges)
        failures_count = np.zeros(self.n_bridges)
        
        u = np.random.uniform(0.01, 0.99, self.n_bridges)
        eta = self.bridges['eta'].values
        beta = self.bridges['beta'].values
        time_to_failure = eta * (-np.log(1.0 - u)) ** (1.0 / beta)
        
        for i in range(self.n_bridges):
            bridge_aadt = self.bridges.loc[i, 'AADT']
            bridge_weather = self.bridges.loc[i, 'freeze_thaw_cycles']
            bridge_eta = eta[i]
            bridge_beta = beta[i]

            cm_val = 50000000
            cf_val = 50000000000
            vot_val = 11000
            w_val = 1.0
            w_max_val = 2.0
            lam_val = 1000000
            alpha_val = 0.002
            
            if ablation_mode == "no_weather":
                alpha_val = 0
            elif ablation_mode == "no_opportunity":
                vot_val = 0
            elif ablation_mode == "no_penalty":
                lam_val = 0

            # 새로운 정책 함수 적용
            if policy_type == "법정":
                t_star = policy_legal(bridge_eta, bridge_beta, bridge_weather, alpha_val)
            elif policy_type == "임계값":
                t_star = policy_threshold(bridge_eta, bridge_beta, threshold_grade=2.0, N_FT=bridge_weather, alpha=alpha_val)
            elif policy_type == "Frangopol":
                t_star = policy_frangopol(bridge_aadt, cm_val, cf_val, bridge_weather, alpha_val)
            else:
                try:
                    t_star = cost_function.optimize_T(
                        AADT=bridge_aadt, Cm=cm_val, Cf=cf_val, VOT=vot_val, 
                        w=w_val, W_max=w_max_val, lam=lam_val, N_FT=bridge_weather, alpha=alpha_val
                    )
                except:
                    t_star = 5.0

            t_curr = 0
            fail_time = time_to_failure[i]
            
            while t_curr < years:
                next_maintenance = t_curr + t_star
                
                if fail_time < next_maintenance and fail_time < years:
                    failures_count[i] += 1
                    total_costs[i] += cost_function.ETC(
                        T=t_star, AADT=bridge_aadt, Cf=cf_val, N_FT=bridge_weather, alpha=alpha_val
                    )
                    t_curr = fail_time
                    fail_time = t_curr + bridge_eta * (-np.log(1.0 - np.random.uniform(0.1, 0.99))) ** (1.0 / bridge_beta)
                elif next_maintenance < years:
                    total_costs[i] += cm_val / t_star
                    t_curr = next_maintenance
                else:
                    break
                    
        return total_costs.sum(), failures_count.mean()
