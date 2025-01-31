def flippingBits(n):
    result = 0
    i = 0
    while n:
        n, m = divmod(n, 2)
        print(n, m)
        if m==0:
            result += (2**i)
        i += 1
        
    return result

print(flippingBits(3))