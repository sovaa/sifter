#!/usr/bin/env python3

import math
import random
import sys

USERS=int(sys.argv[1])
ITEMS=int(sys.argv[2])
MAX_SCORE=10.0
MAX_RATED_ITEMS_PER_USER=200


def should_continue():
    return random.random()*50 < 5

for user in range(USERS):
    for item in range(MAX_RATED_ITEMS_PER_USER):
        if should_continue():
            continue

        item_id = int(math.floor(random.random()*ITEMS))
        score = round(random.random(), 1)
        print("%s\t%s\t%s" % (user, item, score))

