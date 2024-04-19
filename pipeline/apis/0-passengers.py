#!/usr/bin/env python3
"""
Swapi API, create a method that returns the list of
ships that can hold a given number of passengers
"""


import requests


def availableShips(passengerCount):
    """Swapi API, create a method that returns the list of
    ships that can hold a given number of passengers"""
    import requests
    url = "https://swapi-api.alx-tools.com/api/starships/"
    response = requests.get(url)
    data = response.json()
    ships = []
    # print(data["results"])
    for result in data['results']:
        if result['passengers'] != "n/a":
            passengers_no = int(result['passengers'].replace(',', ''))
            # print(passengers_no)
            if passengers_no >= passengerCount:
                ships.append(result['name'])

    return ships
