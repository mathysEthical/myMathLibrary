import mymathlibrary as mml
import math

class Rational:

    def __new__(cls,a,b):
        if(int(str(b.value))!=0 and abs(int(str(a.value))/int(str(b.value)))%1<10**(-10)):
            return mml.ZZ(math.floor(float(a.value/b.value)))
        obj = object.__new__(cls)
        return obj

    def __hash__(self):
        return hash(str(self))

    def __init__(self,a,b):
        if(mml.type(a)!=mml.Integer or mml.type(b)!=mml.Integer):
            raise ValueError(f"numerator and denominator of Rational must be Integer, got: {mml.type(a).__name__} and {type(b).__name__}") 
        self.a=a.value
        self.b=b.value
        # if(abs(round(self.a/self.b)-self.a/self.b)<10**(-5000000000000050000000000000)):
        #     self.a=round(self.a/self.b)
        #     self.b=1
        self.reduce()

    def __round__(self,*args):
        if(len(args)==1):
            digits=args[0]
        else:
            digits=0
        return round(mml.RR(self.a/self.b),digits)

    def reduce(self):
        if(self.b<0):
            self.a*=-1
            self.b*=-1
        reduceFactor=mml.gcd(self.a,self.b)
        self.a//=reduceFactor
        self.b//=reduceFactor

    def __float__(self):
        return float(self.a)/float(self.b)

    def __str__(self) -> str:
        if(self.b==1):
            return f"{self.a}"
        return f"{self.a}/{self.b}"
    
    def __radd__(self,other):
        return self+other

    def __add__(self,other):
        if(mml.type(other)==mml.Integer):
            numerator=self.a+other*self.b
            denominator=self.b
            return QQ(numerator,denominator)
        if(mml.type(other) in [mml.Polynom,mml.Complex,mml.MonovariatePolynom]):
            return other+self
        if(mml.type(other)==mml.Real):
            return self.a/self.b+other
        if(mml.type(other)!=Rational):
            raise NotImplementedError(f"Addition between Rational and {mml.type(other).__name__} is not implemented")
        numerator=self.a*other.b+other.a*self.b
        denominator=self.b*other.b
        return mml.QQ(numerator,denominator)

    def __rsub__(self,other):
        return -(self-other)

    def __sub__(self,other):
        if(mml.type(other) in [mml.Integer]):
            denominator=self.b
            numerator=self.a-other*self.b
            return QQ(numerator,denominator)
        if(mml.type(other) in [mml.Real]):
            return mml.RR(self.a/self.b)-other
        denominator=self.b*other.b
        numerator=self.a*other.b-other.a*self.b
        return mml.QQ(numerator,denominator)

    def __rmul__(self,other):
        return self*other

    def __pow__(self,power):
        if(mml.type(power) in [mml.Integer]):
            return mml.QQ(self.a**power,self.b**power)
        if(mml.type(power) in [mml.Real]):
            return (self.a/self.b)**power
        raise NotImplementedError(f"Exponentiation of Rational with {mml.type(power).__name__} is not implemented")
    def __mul__(self,other):
        if(mml.type(other) in [mml.Matrix,mml.Vector]):
            return other*self
        if(mml.type(other)==mml.Integer):
            return mml.QQ(self.a*other,self.b)
        if(mml.type(other) in [mml.Real,complex,mml.Complex]):
            return self.a/self.b*other 
        if(mml.type(other)==mml.Polynom):
            return other*self
        assert mml.type(other)==mml.Rational, f"Multiplication between Rational and {mml.type(other).__name__} is not implemented"  
        denominator=self.b*other.b
        numerator=self.a*other.a
        return mml.QQ(numerator,denominator)

    def inverse(self):
        return mml.QQ(self.b,self.a)

    def __truediv__(self,other):
        if(mml.type(other) in [Rational]):
            otherInverse=other.inverse()
            return self*otherInverse
        elif(mml.type(other) in [mml.Integer]):
            otherInverse=mml.QQ(1,other)
            return self*otherInverse
        if(mml.type(other) in [mml.Real]):
            return self.a/self.b/other
        raise NotImplementedError(f"Division between Rational and {mml.type(other).__name__} is not implemented")

    def __rtruediv__(self,other):
        return other*self.inverse()
    
    def __abs__(self):
        return mml.QQ(abs(self.a),abs(self.b))

    def __gt__(self,other) -> bool:
        if(mml.type(other) in [Rational]):
            return self.a*other.b>other.a*self.b
        if(mml.type(other) in [mml.Real]):
            return mml.RR(self.a/self.b)>other
        if(mml.type(other) in [mml.Integer]):
            return self.a>other*self.b
        raise NotImplementedError(f"Operator > between Rational and {mml.type(other).__name__} is not implemented")
    def __neg__(self):
        return mml.QQ(-self.a,self.b)

    def __repr__(self):
        return str(self)

    def __lt__(self,other):
        # self < other
        if(mml.type(other)==mml.Integer):
            return (self-other).a<0
        if(mml.type(other) in [mml.Real]):
            return self.a/self.b<other
        return (self-other).a<0

    def __pos__(self):
        return self
    def __rmod__(self,other):
        # other % self
        if(mml.type(other) in [mml.Real]):
            return other%(self.a/self.b)
        if(mml.type(other) in [mml.Integer]):
            other=mml.QQ(other,1)
            return other%self


    def __mod__(self,modulo):
        if(mml.type(modulo) in [mml.Real]):
            return (self.a/self.b)%modulo
        if(mml.type(modulo)==mml.Integer):
            return self-self.a//(self.b*modulo)*modulo
        if(mml.type(modulo) in [Rational]):
            return self-(self//modulo)*modulo
        else:
            assert False, f"Modulo operator between Rational and {mml.type(modulo).__name__} is not implemented"

    def __floordiv__(self,other):
        if(mml.type(other) in [Rational]):
            return mml.ZZ((self.a*other.b)//(other.a*self.b))
        return mml.ZZ((self.a/self.b)//other)

    def __eq__(self,other):
        if(mml.type(other)==mml.Rational):
            return self.a==other.a and self.b==other.b
        if(mml.type(other)==mml.Integer):
            return self.b==1 and self.a==other
        if(mml.type(other)==mml.Real):
            return self.a/self.b==other
