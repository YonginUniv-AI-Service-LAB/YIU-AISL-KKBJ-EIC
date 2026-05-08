import numpy as np
import pandas as pd
import uuid

def generate_bridge_data(n_bridges=200):
    regions = [
        '서울', '경기', '강원', '인천', '부산', '대구', 
        '대전', '광주', '울산', '세종', '충북', '충남', 
        '전북', '전남', '경북', '경남', '제주'
    ]
    
    region_aadt_means = {
        '서울': 102000, 
        '경기': 40677,  
        '강원': 8292,   
        '인천': 35000,  
        '부산': 32000,  
        '경남': 15000,  
        '경북': 12000,  
        '충남': 14000,  
        '충북': 11000,  
        '전남': 9000,   
        '전북': 9500,   
        '대구': 28000,  
        '대전': 25000,  
        '광주': 22000,  
        '울산': 24000,  
        '세종': 18000,  
        '제주': 7000    
    }

    region_nft_means = {
        '강원': 80, '경북': 65, '충북': 60, '경기': 55, '서울': 50,
        '전북': 45, '충남': 45, '대전': 40, '세종': 40, '대구': 35,
        '인천': 35, '광주': 30, '경남': 25, '울산': 25, '전남': 20,
        '부산': 15, '제주': 10
    }

    bridges = []
    
    for _ in range(n_bridges):
        region = np.random.choice(regions)
        
        aadt_mean = region_aadt_means[region]
        aadt = max(500, int(np.random.normal(aadt_mean, aadt_mean * 0.2)))
        
        nft_mean = region_nft_means[region]
        nft = max(0, int(np.random.normal(nft_mean, 5)))
        
        bridge = {
            'bridge_id': str(uuid.uuid4())[:8],
            'bridge_name': f"{region}_Bridge_{np.random.randint(100, 999)}",
            'region': region,
            'aadt': aadt,
            'n_ft': nft,
            'years_since_built': np.random.randint(30, 60),
            'base_eta': np.random.uniform(20, 30),
            'beta': np.random.uniform(2.0, 3.5)
        }
        bridges.append(bridge)
    
    df = pd.DataFrame(bridges)
    return df

if __name__ == "__main__":
    bridge_df = generate_bridge_data(200)
    bridge_df.to_csv("simulated_bridges.csv", index=False, encoding='utf-8-sig')
    print(bridge_df.groupby('region')['aadt'].mean().sort_values(ascending=False))