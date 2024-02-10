def fact(num):
    """ Calculates num!"""
    values = [num for num in range(1, num+1)]
    factorial = 1
    if num == 0:
        return 1
    for value in values:
        factorial *= value
    return factorial

def comb(n, r):
    """ calculates the combination nCr"""
    return fact(n) / (fact(n - r) * fact(r))

print(fact(4))
print(comb(4, 2))