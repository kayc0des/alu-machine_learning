#!/usr/bin/env python3
'''
Create a method that returns the list
of ships that can hold a given # of passengers
'''

import requests


def availableShips(passengerCount):
    ''' Returns the list of ships that can
    hold a given number of passengers

    passengerCount: number of passengers
    '''
    if not isinstance(passengerCount, int):
        raise TypeError('passengerCount must be an integer')

    ship_list = []

    try:
        url = 'https://swapi-api.alx-tools.com/api/starships/'
        while url:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            ships = data['results']

            for ship in ships:
                if ship['passengers'] == 'n/a':
                    continue
                try:
                    if int(ship['passengers']
                           .replace(',', '')) >= passengerCount:
                        ship_list.append(ship['name'])
                except ValueError:
                    continue
            # Get the next page URL
            url = data.get('next')

        return ship_list

    except requests.RequestException as e:
        print("Request error: {}".format(e))
        return []
    except Exception as err:
        print("An error occurred: {}".format(err))
        return []
