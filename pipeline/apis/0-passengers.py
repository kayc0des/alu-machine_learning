#!/usr/bin/env python3
'''

'''

import requests

def availableShips(passengerCount):
    ''' Returns the list of ships that can
    hold a given number of passengers

    passengerCount: number of passengers
    '''
    try:
        url = f'https://swapi-api.alx-tools.com/api/people/{passengerCount}/'
        response = requests.get(url)
        print(response.status_code)
        
        return response
    except Exception as err:
        print(err)
        
        
if __name__ == '__main__':
    ship = availableShips(4)
    print(ship.json())