import requests

url = 'https://api.github.com/users/kayc0des'

response = requests.get(url)

data = response.json()

print(data['location'])