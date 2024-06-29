import requests

url = 'https://swapi-api.alx-tools.com/api/species/'

response = requests.get(url)

data = response.json()

species = data['results']

print(species[0])