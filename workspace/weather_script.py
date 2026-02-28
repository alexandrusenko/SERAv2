import requests
url = 'https://yandex.ru/pogoda/ru/moscow?lat=55.936752&lon=37.361007'
response = requests.get(url)
print(response.text)