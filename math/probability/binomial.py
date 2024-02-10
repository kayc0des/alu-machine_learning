#!/usr/bin/env python3
'''
    The Binomial Distribution
'''


class Binomial():
    """ This class represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ initialization method """
        if data is None:
            if n < 1:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = round(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            summation = 0
            for x in data:
                summation += ((x - mean) ** 2)
            variance = summation / len(data)
            q = variance / mean
            p = 1 - q
            n = round(mean / p)
            p = float(mean / n)
            self.n = n
            self.p = p

    def fact(self, num):
        """ Calculates num!"""
        values = [num for num in range(1, num+1)]
        factorial = 1
        if num == 0:
            return 1
        for value in values:
            factorial *= value
        return factorial

    def comb(self, n, r):
        """ calculates the combination nCr"""
        return self.fact(n) / (self.fact(n - r) * self.fact(r))

    def pmf(self, k):
        """ calculates pmf """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        q = 1 - self.p
        combination = self.comb(self.n, k)
        return combination * (self.p ** k) * (q ** (self.n-k))

    def cdf(self, k):
        """ calculates the cdf """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
