from mymathlibrary import *

x=QQ["x"]
A=Matrix([5,-4,4],[4,-3,4],[2,-2,3])

print((A-x*A.Id).det().factor())
print(RR(0)==0)

print(((x-1)*(x-2)).factor())
print(gcd(104,8))
print(crt([10,2],[101,5]))
myRing=modularRing(5)
print(myRing(2)**-1)