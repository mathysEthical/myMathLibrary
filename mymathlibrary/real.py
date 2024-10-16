import mymathlibrary as mml


class Real:
    def __new__(cls,value):
        obj = object.__new__(cls)
        if(type(value)==complex):
            return value
        if(abs(value-round(value,0))<10**(-6)):
            return mml.ZZ(value)
        return obj

    def __init__(self, value: float) -> None:
        self.value=value
    
    def __str__(self):
        return f"{self.value}"
    
    def __repr__(self):
        return str(self)

    def __mod__(self,modulo):
        return self.value%modulo

    def __radd__(self,other):
        return self+other
    def __add__(self,other):
        if(type(other) in [Real,mml.Integer]):
            return mml.RR(self.value+other.value)
        if(type(other) in [mml.MonovariatePolynom]):
            return other+self
        raise NotImplementedError(f"Addition between Real and {type(other).__name__} is not implemented")
    def __float__(self):
        return float(self.value)

    def __rtruediv__(self,other):
        # other/self
        if(type(other) in [mml.Real,mml.Integer]):
            return mml.RR(other.value*(1/self.value))

        raise NotImplementedError(f"Division between {type(other).__name__} and Real is not implemented")
    def __truediv__(self,other):
        if(type(other) in [Real,mml.Integer]):
            return mml.RR(self.value/other.value)
        if(type(other) in [mml.Rational]):
            return mml.RR(self.value*other.b/other.a)
        raise NotImplementedError(f"Division between Real and {type(other).__name__} is not implemented")
    def __neg__(self):
        return mml.RR(-self.value)

    def __rmul__(self,other):
        return RR(self*other)

    def __round__(self,digits):
        return round(self.value,digits)

    def __lt__(self,other) -> bool:
        return self.value<other.value

    def __mul__(self,other):
        if(type(other) in [mml.Rational,mml.Vector]):
            return other*self
        if(type(other) in [Real,mml.Integer]):
            return mml.RR(self.value*other.value)
        if(type(other) in [mml.Matrix]):
            return other*self
        raise NotImplementedError(f"Multiplication between Real and {type(other).__name__} is not implemented")
    def __rsub__(self,other):
        #other-self
        return mml.RR(other.value-self.value)
    def __sub__(self,other):
        return mml.RR(self.value-other.value)

    def __eq__(self,other):
        return self.value==other

    def __abs__(self):
        return abs(self.value)

    def __pow__(self,power):
        return mml.RR(self.value)**float(power)
    def __hash__(self):
        return hash(self.value)
    def __gt__(self,other):
        if(type(other) in [Real]):
            return self.value>other.value
        raise NotImplementedError(f"Operator > between Real and {type(other).__name__} is not implemented")
