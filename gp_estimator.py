import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

def gp_estimate(inspection_years, condition_grades, future_years):
    
    X  = np.array(inspection_years).reshape(-1, 1)
    y  = np.array(condition_grades).ravel()
    Xf = np.array(future_years).reshape(-1, 1)
    
    kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=10.0, nu=2.5)
        + WhiteKernel(noise_level=0.1)
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
    gp.fit(X, y)
    mean, std = gp.predict(Xf, return_std=True)
    return mean, std

def weibull_cdf(T, eta, beta):
    return 1 - np.exp(-(T/eta)**beta)

def estimate_weibull_params(mean_grades, future_years_flat):
    
    years = np.array(future_years_flat).ravel()
    
    mean_grades_clipped = np.clip(mean_grades, 1.0, 5.0)
    failure_probs = (5 - mean_grades_clipped) / 4
    failure_probs = np.clip(failure_probs, 0.01, 0.98)
    
    try:
        params, _ = curve_fit(
            weibull_cdf, years, failure_probs,
            p0=[30, 2.5],
            bounds=([5, 1.5], [100, 10])
        )
        eta, beta = params
        return eta, beta
    except RuntimeError as e:
        print(f"[GP] curve_fit 수렴 실패 — 기본값 사용: {e}")
        return 30.0, 2.5
    except ValueError as e:
        print(f"[GP] curve_fit 입력값 오류 — 기본값 사용: {e}")
        return 30.0, 2.5