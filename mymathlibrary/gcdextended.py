import mymathlibrary as mml


def gcdExtended(a, b):
    global x, y
    # Base Case
    if (a == 0):
        x = 0
        y = 1
        return b

    # To store results of recursive call
    gcd = gcdExtended(b % a, a)
    x1 = x
    y1 = y
    # Update x and y using results of recursive
    # call
    x = y1 - (b // a) * x1
    y = x1
    return gcd
