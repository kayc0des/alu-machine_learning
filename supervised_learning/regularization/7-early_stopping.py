#!/usr/bin/env python3
'''
Determines if we should stop gradient descent
'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''
    Determined if gradient descent should be stopped

    Param:
    cost -> current validation cost
    opt_cost -> is the lowest recorded validation of network
    threshold -> patience count used for stoping
    patience -> patience count
    count -> is the count of how long the threshold
    hasn't been met

    Returns:
    Returns a bool of whether the
    network should be stopped early
    '''

    # Check if the cost has decreased by more than the threshold
    if cost < opt_cost - threshold:
        # Reset count if the validation cost has decreased sufficiently
        return False, 0
    else:
        # Increment count if the validation cost has not decreased
        count += 1
        # Check if the count has reached the patience limit
        if count >= patience:
            return True, count
        else:
            return False, count
