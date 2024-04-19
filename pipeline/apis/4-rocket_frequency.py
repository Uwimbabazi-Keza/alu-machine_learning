#!/usr/bin/env python3

"""
using the (unofficial) SpaceX API, write a script that displays
the number of launches per rocket
"""

import requests


def launches_per_rocket():
    """using the (unofficial) SpaceX API, write a script that displays
    the number of launches per rocket"""
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    data = response.json()
    rockets = {}
    for launch in data:
        rocket_id = launch['rocket']
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(
            rocket_id)
        rocket_response = requests.get(rocket_url)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data['name']
        if rocket_name in rockets:
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1
    for rocket in sorted(rockets.items(), key=lambda x: (-x[1], x[0])):
        print("{}: {}".format(rocket[0], rocket[1]))


if __name__ == "__main__":
    launches_per_rocket()
