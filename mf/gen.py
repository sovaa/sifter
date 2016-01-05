#!/usr/bin/env python3

import math
import random

USERS=3
ITEMS=10
MAX_SCORE=10.0

def should_continue():
    return random.random()*random.random()*5 > 1

for user in range(USERS):
    for item in range(ITEMS):
        if should_continue():
            continue

        score = round(random.random(), 1)
        print("%s\t%s\t%s" % (user, item, score))

