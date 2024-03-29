#!/usr/bin/env python3
"""A method that returns the list of ships
that can hold a given number of passengers"""

import requests

def availableShips(passengerCount):
    """A method that returns the list of ships
    that can hold a given number of passengers"""

    url = "https://swapi-api.hbtn.io/api/starships/?format=json"
    
    ships = []
    
    while url:
        r = requests.get(url)
        data = r.json()
        starships = data.get("results")
        for starship in starships:
            if starship.get('passengers').isdigit():
                if int(starship['passengers']) >= passengerCount:
                    ships.append(starship['name'])
        url = data.get("next")
    
    return ships
