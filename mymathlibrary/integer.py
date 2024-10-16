import mymathlibrary as mml
import math

class Integer:
    def __init__(self,value: int):
        if(mml.type(value)==bool):
            value=int(value)
        
        self.value=int(str(math.floor(round(value,0))))
    def __str__(self):
        return f"{self.value}"
    def __repr__(self):
        return str(self)
    def __radd__(self,other):
        return self+other

    def __hash__(self):
        return hash(self.value)

    def __rfloordiv__(self,other):
        if(mml.type(other)==Integer):
            return self.value//other.value
        raise NotImplementedError(f"Floor division between {type(other).__name__} and Integer is not implemented")
    def __float__(self):
        return float(self.value)

    def __round__(self,*args):

        digits=0
        if(len(args)==1):
            digits=args[0]
        return self

    def __neg__(self):
        return mml.ZZ(-self.value)

    def __add__(self,other):
        if(mml.type(other)==Integer):
            return mml.ZZ(self.value+other.value)
        if(mml.type(other)==mml.Rational):
            return mml.QQ(self.value*other.b+other.a,other.b)
        if(mml.type(other) in [complex,mml.Real]):
            return self.value+other
        if(mml.type(other) in [mml.Polynom,mml.MonovariatePolynom]):
            return other+self
        raise ValueError(f"Addition not implemented between Integer and {mml.type(other).__name__}")
    def __rmul__(self,other):
        return self*other
    def __mul__(self,other):
        if(mml.type(other)==Integer or mml.type(other)==int):
            return mml.ZZ(self.value*other.value)
        if(mml.type(other)==mml.Real):
            return other*self.value
        if(mml.type(other)==mml.Rational):
            return mml.QQ(self.value*other.a,other.b)
        if(mml.type(other).__name__ in ["Matrix","Polynom","Vector","MonovariatePolynom"]):
            return other*self
        if(mml.type(other)==complex):
            return self.value*other
        raise NotImplementedError(f"Multiplication between Integer and {mml.type(other).__name__} is not implemented")
    def __floordiv__(self,other):
        return mml.ZZ(self.value//other)
    
    def __rsub__(self,other):
        return self+other*-1

    def __rtruediv__(self,other):
        # other/self
        if(mml.type(other)==Integer):
            return mml.QQ(other,self)
        else:
            return other/self
    def __abs__(self):
        return mml.ZZ(abs(self.value))
    def __mod__(self,other):
        if(mml.type(other)==Integer):
            return mml.ZZ(self.value%other.value)
        else:
            return self.value%other
    
    
    def __sub__(self,other):
        return self+other*-1

    def __eq__(self,other):
        if(mml.type(other)==mml.Real):
            return self.value==other
        if(mml.type(other)==mml.Rational):
            return mml.QQ(self.value,1)==other
        if(mml.type(other)==mml.Integer):
            return self.value==other.value
        ValueError(f"Equality between Integer and {mml.type(other)} is not implemented")
    
    def __pow__(self,power):
        if(mml.type(power)==Integer):
            return mml.ZZ(self.value**power)
        if(mml.type(power) in [mml.Rational,mml.Real]):
            return mml.RR(self.value**power)
        raise NotImplementedError(f"Exponentiation between Integer and {type(power).__name__} is not implemented")
    def __lt__(self,other):
        if(type(other)==int):
            other=mml.ZZ(other)
        return self.value<other.value

    def __ge__(self,other):
        return self>other or self==other
    def __le__(self,other):
        return self<other or self==other

    def __gt__(self,other):
        if(mml.type(other)==int):
            other=mml.ZZ(other)
        return self.value>other.value

    def __truediv__(self,other):
        if(mml.type(other) in [Integer]):
            return mml.QQ(self,other)
        if(mml.type(other) in [mml.Real,complex]):
            return self.value/other
        if(mml.type(other) in [mml.Rational]):
            return self*mml.QQ(other.b,other.a)
        raise ValueError(f"Division between Integer and {type(other).__name__} is not implemented")
