import numpy as np
import cost_function

class BridgeSimulator:
    def __init__(self, bridges_df):
        self.bridges = bridges_df
        self.n_bridges = len(bridges_df)
        
    def simulate_policy(self, policy_type="제안", ablation_mode=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        years = 30
        total_costs = np.zeros(self.n_bridges)
        failures_count = np.zeros(self.n_bridges)
        cm_val, cf_val, vot_val, alpha_val, lam_val = 50000000, 50000000000, 11000, 0.002, 1000000
        if ablation_mode == "no_weather": alpha_val = 0
        elif ablation_mode == "no_opportunity": vot_val = 0
        elif ablation_mode == "no_penalty": lam_val = 0
        for i in range(self.n_bridges):
            b = self.bridges.iloc[i]
            if policy_type == "법정":
                t_star = 2.0
            elif policy_type == "임계값":
                t_star = 5.0
            elif policy_type == "Frangopol":
                t_star = 7.5
            else:
                t_star = cost_function.optimize_T(
                    AADT=b['AADT'], Cm=cm_val, Cf=cf_val, VOT=vot_val,
                    N_FT=b['freeze_thaw_cycles'], alpha=alpha_val, lam=lam_val
                )
            annual_cost = cost_function.ETC(
                T=t_star, AADT=b['AADT'], Cm=cm_val, Cf=cf_val, VOT=vot_val,
                N_FT=b['freeze_thaw_cycles'], alpha=alpha_val, lam=lam_val
            )
            total_costs[i] = annual_cost * years
            prob = cost_function.failure_prob(t_star, eta=b['eta'], beta=b['beta'], 
                                               N_FT=b['freeze_thaw_cycles'], alpha=alpha_val)
            failures_count[i] = prob * (years / t_star)
        return total_costs.mean(), failures_count.mean()

    def run_monte_carlo(self, policy_type="제안", ablation_mode=None, iter_num=100):
        costs, failures = [], []
        for run in range(iter_num):
            cost, fail = self.simulate_policy(policy_type, ablation_mode, seed=run)
            costs.append(cost)
            failures.append(fail)
        return np.mean(costs), np.std(costs), np.mean(failures)