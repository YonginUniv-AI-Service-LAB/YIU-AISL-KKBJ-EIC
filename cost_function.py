# 비용 계산 함수 (와이블, ETC, T*)
import numpy as np
from scipy.optimize import minimize_scalar

# 단위비용 상수 (KDI 표준지침 표 Ⅵ-3, Ⅵ-6 기반)
VOT_DEFAULT = 11000   # 통행시간가치 (원/시간), KDI 표 Ⅵ-6 평균값
VOC_DEFAULT = 250     # 차량운행비용 (원/km),  KDI 표 Ⅵ-3 승용차 기준
D_MAINT     = 3       # 정비 시 교통 통제 일수 (일) — 민감도 분석 대상
DELTA_T     = 0.1     # 우회 시 추가 통행시간 (시간) — 민감도 분석 대상
DELTA_D     = 1.0     # 우회 시 추가 거리 (km) — 민감도 분석 대상
D_COLLAPSE   = 30           # 붕괴 시 통제 일수 (가정값)  — 민감도 분석 대상
C_EMERGENCY  = 500000000  # 긴급 보수비 (원, 가정값)    — 민감도 분석 대상

def calc_cf(AADT,
            VOT=VOT_DEFAULT,
            VOC=VOC_DEFAULT,
            D_collapse=D_COLLAPSE,
            c_emergency=C_EMERGENCY):
    return c_emergency + AADT * D_collapse * (VOT + VOC)

def failure_prob(T, eta=30, beta=2.5, N_FT=30, alpha=0.002):
    eta_local = eta * np.exp(-alpha * N_FT)
    return 1 - np.exp(-(T/eta_local)**beta)

def labor_load(T):
    return 6.0 / T

def ETC(T, eta=30, beta=2.5,
        AADT=50000,
        C_m=300000000,
        C_f=None,
        VOT=VOT_DEFAULT, 
        VOC=VOC_DEFAULT,
        W_max=2.0,
        lam=1000000,
        N_FT=30,
        alpha=0.002,
        use_user_cost=True,
        use_labor_penalty=True):
    
    # 1.정비비 (단위 시간당)
    maintenance_cost = C_m / T
    
    # 2.기대 고장비
    if C_f is None:
        C_f = calc_cf(AADT, VOT=VOT, VOC=VOC)
    failure_cost = C_f * failure_prob(T, eta=eta, beta=beta,
                                      N_FT=N_FT, alpha=alpha) / T
    
    # 3.사용자 기회비용 (정비 시 우회 발생분)
    if use_user_cost:
        C_U = AADT * D_MAINT * (VOT * DELTA_T + VOC * DELTA_D)
        user_opportunity_cost = C_U / T
    else:
        user_opportunity_cost = 0
    
    # 4.인력 페널티
    labor_penalty = lam * max(0, labor_load(T) - W_max) if use_labor_penalty else 0
    return maintenance_cost + failure_cost + user_opportunity_cost + labor_penalty

def optimize_T(eta=30, beta=2.5,
               AADT=50000,
               C_m=300000000,
               C_f=None,
               VOT=VOT_DEFAULT,
               VOC=VOC_DEFAULT,
               W_max=2.0,
               lam=1000000,
               N_FT=30,
               alpha=0.002,
               use_user_cost=True,
               use_labor_penalty=True):
    
    result = minimize_scalar(
        lambda T: ETC(T, eta=eta, beta=beta,
                      AADT=AADT, C_m=C_m, C_f=C_f,
                      VOT=VOT, VOC=VOC,
                      W_max=W_max, lam=lam,
                      N_FT=N_FT, alpha=alpha,
                      use_user_cost=use_user_cost,
                      use_labor_penalty=use_labor_penalty),
        bounds=(1, 60),
        method='bounded'
    )
    return result.x

def calc_lcc_savings(T_star,
                     eta=30, beta=2.5,
                     AADT=50000,
                     C_m=300000000,
                     C_f=None,
                     VOT=VOT_DEFAULT,
                     VOC=VOC_DEFAULT,
                     N_FT=30,
                     alpha=0.002,
                     years=30):
    
    T_legal = 2  # 법정 주기 (등급 B·C 기준 2년)
    
    cost_legal = ETC(T_legal, eta=eta, beta=beta,
                     AADT=AADT, C_m=C_m, C_f=C_f,
                     VOT=VOT, VOC=VOC,
                     N_FT=N_FT, alpha=alpha) * years
    
    cost_optimized = ETC(T_star, eta=eta, beta=beta,
                         AADT=AADT, C_m=C_m, C_f=C_f,
                         VOT=VOT, VOC=VOC,
                         N_FT=N_FT, alpha=alpha) * years
    
    savings = (cost_legal - cost_optimized) / cost_legal * 100
    return savings