#!/usr/bin/env python3
'''
Using GitHub API
write a script that prints the location
of a specific user
'''


import requests
import time
from datetime import datetime


def user_loc(url):
    '''
    Prints the location of a specific user
    '''

    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = int(response.headers["X-Ratelimit-Reset"])
        current_timestamp = int(time.time())
        reset_in_minutes = (reset_timestamp - current_timestamp) // 60
        print("Reset in {} min".format(reset_in_minutes))
    else:
        print(response.json()["location"])

if __name__ == "__main__":
    import sys
    user_loc(sys.argv[1])
