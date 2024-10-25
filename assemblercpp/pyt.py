# import package
import requests


#資料網址
api_key="06e299bf-c699-43fd-832d-c1f98fb8413a"
url="https://data.moenv.gov.tw/api/v2/"
datan="aqx_p_432"
ext="?api_key="
aqi_url=url+datan+ext+api_key


#取出測站資料
aqi = requests.get(aqi_url).json()
print(aqi)