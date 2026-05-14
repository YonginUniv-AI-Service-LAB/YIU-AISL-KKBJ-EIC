import numpy as np
from scipy import stats

def perform_statistical_tests(results_dict):
    legal = results_dict["법정"]
    threshold = results_dict["임계값"]
    frangopol = results_dict["Frangopol"]
    proposed = results_dict["제안"]
    
    stat, p_friedman = stats.friedmanchisquare(legal, threshold, frangopol, proposed)
    
    p_values_post = {}
    for name, data in [("법정", legal), ("임계값", threshold), ("Frangopol", frangopol)]:
        _, p_val = stats.wilcoxon(proposed, data)
        p_values_post[name] = min(1.0, p_val * 3)
        
    return p_friedman, p_values_post

def run_sensitivity_analysis(simulator, param_name, variations=[-0.5, 0.0, 0.5]):
    sensitivity_results = {}
    for var in variations:
        mean_cost, _, _ = simulator.run_monte_carlo(policy_type="제안", iter_num=10)
        sensitivity_results[f"{param_name}_{int(var*100):+}%"] = mean_cost
        
    return sensitivity_results