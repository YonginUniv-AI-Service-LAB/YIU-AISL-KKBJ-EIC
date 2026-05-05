#GP 점검 등급 예측
import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

def gp_estimate(inspection_years, condition_grades, future_years):
    kernel = ConstantKernel(1.0) * Matern(length_scale=10.0, nu=2.5) \
           + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
    gp.fit(inspection_years, condition_grades)
    mean, std = gp.predict(future_years, return_std=True)
    return mean, std

def weibull_cdf(T, eta, beta):
    return 1 - np.exp(-(T/eta)**beta)

def estimate_weibull_params(mean_grades, future_years_flat):
    # 등급을 고장확률로 변환 (A=5 → 0%, E=1 → 100%)
    failure_probs = (5 - mean_grades) / 4
    failure_probs = np.clip(failure_probs, 0.01, 0.99)
    
    try:
        params, _ = curve_fit(weibull_cdf, future_years_flat, failure_probs,
                             p0=[30, 2.5], bounds=([1, 0.5], [100, 10]))
        eta, beta = params
        return eta, beta
    except:
        return 30, 2.5