#!/usr/bin/env python3
'''
    The Binomial Distribution
'''


class Binomial():
    """ This class represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ initialization method """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a postive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = round(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate p by counting successes
            successes = sum(1 for x in data if x == 1)
            p = successes / len(data)
            # if p < 0 or p > 1:
            #     raise ValueError("Invalid probability in data")
            # Calculate n using the formula n = sum(data) / p
            self.n = round(sum(data) / p)
            # Recalculate p based on the rounded n
            self.p = successes / self.n
