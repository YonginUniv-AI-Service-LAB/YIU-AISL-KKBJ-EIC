import numpy as np
import pandas as pd

def generate_synthetic_bridges(n=200, seed=42):
    np.random.seed(seed)
    
    regions = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]
    
    region_weather_means = {
        "서울": 15, "인천": 14, "경기": 18, "강원": 28, "충북": 22, 
        "충남": 16, "경북": 20, "전북": 15, "전남": 8, "경남": 10, 
        "부산": 5, "대구": 12, "울산": 7, "광주": 9, "대전": 14, 
        "세종": 15, "제주": 2
    }
    
    region_traffic_means = {
        "서울": 55000, "경기": 45000, "인천": 40000, "부산": 35000,
        "대구": 28000, "대전": 27000, "광주": 24000, "울산": 25000,
        "세종": 20000, "충남": 22000, "충북": 19000, "경남": 21000,
        "경북": 18000, "전북": 16000, "전남": 14000, "강원": 15000, "제주": 12000
    }
    
    bridges = []
    
    for i in range(n):
        region = np.random.choice(regions)
        
        weather_cycle = max(1, int(np.random.normal(region_weather_means[region], 3)))
        aadt = max(1000, int(np.random.normal(region_traffic_means[region], 5000)))
        
        eta = np.random.uniform(25.0, 45.0)
        beta = np.random.uniform(2.0, 3.5)
        
        bridges.append({
            "bridge_id": i,
            "region": region,
            "AADT": aadt,
            "freeze_thaw_cycles": weather_cycle,
            "eta": eta,
            "beta": beta
        })
        
    return pd.DataFrame(bridges)

if __name__ == "__main__":
    df = generate_synthetic_bridges(200)
    print(df.head())
    print("\n교량 생성기 테스트 성공!")