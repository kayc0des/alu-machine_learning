#!/usr/bin/env python3
'''
    Exponential class
'''


class Exponential():
    """ Represents an exponential distribution """

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """ calculates the probability density function """
        # assigning the value of euler's number to a variable e
        e = 2.7182818285
        if x < 0:
            return 0
        return self.lambtha * (e ** (-(self.lambtha) * x))
        