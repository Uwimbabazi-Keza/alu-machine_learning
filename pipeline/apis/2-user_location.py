#!/usr/bin/env python3

"""
using the GitHub API, write a script
that prints the location of a specific user"""

import requests
import sys
import time


def print_location():
    """using the GitHub API, write a script
    that prints the location of a specific user"""
    url = sys.argv[1]
    response = requests.get(url)
    data = response.json()

    if response.status_code == 403:
        rate_limit = int(response.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))

    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 200:
        print(data['location'])


if __name__ == "__main__":
    print_location()
