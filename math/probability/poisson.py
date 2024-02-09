#!/usr/bin/env python3
'''
    Create a class that represents the poisson distribution
'''

class Poisson():
    """ This class represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ initialization method """
        if not data:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))
