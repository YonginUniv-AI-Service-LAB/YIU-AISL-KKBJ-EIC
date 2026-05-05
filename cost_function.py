# 비용 계산 함수 (와이블, ETC, T*)
import numpy as np
from scipy.optimize import minimize_scalar

def failure_prob(T, eta=30, beta=2.5, N_FT=30, alpha=0.002):
    eta_local = eta * np.exp(-alpha * N_FT)
    return 1 - np.exp(-(T/eta_local)**beta)

def ETC(T, AADT=50000, Cm=50000000, Cf=50000000000, VOT=11000, w=1.0, W_max=2.0, lam=1000000, N_FT=30, alpha=0.002):
    정비비 = Cm / T
    고장비 = failure_prob(T, N_FT=N_FT, alpha=alpha) * Cf / T
    사용자우회비용 = AADT * 3 * VOT / T
    인력페널티 = lam * max(0, w - W_max)
    return 정비비 + 고장비 + 사용자우회비용 + 인력페널티

def optimize_T(AADT=50000, Cm=50000000, Cf=50000000000, VOT=11000, w=1.0, W_max=2.0, lam=1000000, N_FT=30, alpha=0.002):
    result = minimize_scalar(
        lambda T: ETC(T, AADT, Cm, Cf, VOT, w, W_max, lam, N_FT, alpha),
        bounds=(1, 60), method='bounded'
    )
    return result.x

def calc_lcc_savings(T_star, AADT=50000, Cm=50000000, Cf=50000000000,
                     VOT=11000, N_FT=30, alpha=0.002, years=30):
    """현행 법정 주기 대비 30년 LCC 절감률 계산"""
    T_legal = 2  # 법정 주기 (등급 B·C 기준 2년)
    
    cost_legal = ETC(T_legal, AADT, Cm, Cf, VOT, N_FT=N_FT, alpha=alpha) * years
    cost_optimized = ETC(T_star, AADT, Cm, Cf, VOT, N_FT=N_FT, alpha=alpha) * years
    
    savings = (cost_legal - cost_optimized) / cost_legal * 100
    return savings