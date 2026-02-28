from duckduckgo_search import DDGS
def get_weather(city):
    with DDGS() as ddgs:
        results = ddgs.weather(city)
        return results
if __name__ == '__main__':
    weather_in_moscow = get_weather('Москва')
    print(weather_in_moscow)