# 외부 데이터 자동 수집기
import requests
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# 더미 데이터 (API 막힐 때 대체용)
DUMMY_AADT = {
    "서울": 102000, "강원": 8292, "제주": 20000,
    "부산": 45000, "대구": 35000, "인천": 40000,
    "광주": 30000, "대전": 32000, "울산": 28000,
    "경기": 40677, "충북": 22000, "충남": 25000,
    "전북": 20000, "전남": 18000, "경북": 24000,
    "경남": 27000, "세종": 23000,
}

DUMMY_N_FT = {
    "서울": 30, "강원": 80, "제주": 10,
    "부산": 20, "대구": 25, "인천": 28,
    "광주": 15, "대전": 27, "울산": 22,
    "경기": 32, "충북": 35, "충남": 28,
    "전북": 18, "전남": 12, "경북": 40,
    "경남": 20, "세종": 27,
}

STN_IDS = {
    "서울": 108, "강원": 105, "제주": 184,
    "부산": 159, "대구": 143, "인천": 112,
    "광주": 156, "대전": 133, "울산": 152,
    "경기": 119, "충북": 131, "충남": 129,
    "전북": 146, "전남": 168, "경북": 136,
    "경남": 155, "세종": 239,
}

def get_aadt(region):
    return DUMMY_AADT.get(region, 30000)

def get_freeze_thaw(region):
    if WEATHER_API_KEY is None:
        return DUMMY_N_FT.get(region, 30)
    
    try:
        stn_id = STN_IDS.get(region, 108)
        decoded_key = urllib.parse.unquote(WEATHER_API_KEY)
        url = f"https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey={decoded_key}&pageNo=1&numOfRows=365&dataType=JSON&dataCd=ASOS&dateCd=DAY&startDt=20230101&endDt=20231231&stnIds={stn_id}"
        response = requests.get(url, timeout=10)
        data = response.json()
        temps = [float(item["minTa"]) for item in data["response"]["body"]["items"]["item"] if item["minTa"] != ""]
        N_FT = sum(1 for i in range(1, len(temps)) if temps[i-1] < 0 and temps[i] >= 0)
        return N_FT
    except Exception as e:
        print(f"[fallback] 기상청 API 실패: {e}")
        return DUMMY_N_FT.get(region, 30)