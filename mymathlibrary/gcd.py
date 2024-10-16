import mymathlibrary as mml

def gcd(a, b):
#   if(a==0 or b==0):return 1
  a=abs(a)
  b=abs(b)
  while ((a % b) > 0):
    R = a % b
    a = b
    b = R
  return b