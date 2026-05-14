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

            if policy_type == "법정":
                t_star = 2.0
            elif policy_type == "임계값":
                t_star = 5.0
            elif policy_type == "Frangopol":
                t_star = 7.5
            else:
                try:
                    t_star = cost_function.optimize_T(
                        AADT=bridge_aadt, 
                        Cm=cm_val, 
                        Cf=cf_val, 
                        VOT=vot_val, 
                        w=w_val, 
                        W_max=w_max_val, 
                        lam=lam_val, 
                        N_FT=bridge_weather, 
                        alpha=alpha_val
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

    def run_monte_carlo(self, policy_type="제안", ablation_mode=None, iter_num=100):
        costs = []
        failures = []
        
        for run in range(iter_num):
            cost, fail = self.simulate_policy(policy_type, ablation_mode, seed=run)
            costs.append(cost)
            failures.append(fail)
            
        return np.mean(costs), np.std(costs), np.mean(failures)