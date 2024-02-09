#!/usr/bin/env python3
'''
    Create a class that represents the poisson distribution
'''


class Poisson():
    """ This class represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ initialization method """
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
                self.lambtha = float(sum(data) / len(data))

    def factorial(self, num):
        """ Calculates num!"""
        if not isinstance(num, int):
            raise TypeError("Number must be an integer")
        else:
            return num * self.factorial(num-1)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes"""
        # assigning the value of euler's number to a variable e
        e = 2.71828
        try:
            if not isinstance(k, int):
                k = int(k)
                return (e**-self.lambtha) * self.lambtha**k / self.factorial(k)
        except ValueError:
            return 0
