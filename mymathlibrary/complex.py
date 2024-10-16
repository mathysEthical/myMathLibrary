import mymathlibrary as mml


class Complex:
    def __init__(self,re,im) -> None:
        self.re=re
        self.im=im
    def __str__(self):
        if(self.im==0):
            return f"{self.re}"
        if(self==i):
            return "i"
        if(self==-i):
            return "-i"
        if(self.re==0):
            return f"{self.im}*i"
        if(self.im==1):
            return f"{self.re}+i"
        if(self.im==-1):
            return f"{self.re}-i"
        return f"{self.re}+{self.im}*i"

    def __neg__(self):
        return self*-1

    def __rmul__(self,other):
        return self*other

    def __abs__(self):
        return distance(Point(0,0),Point(self.re,self.im))

    def __rsub__(self,other):
        # other-self
        return other+self*-1

    def __sub__(self,other):
        return self+other*-1

    def __radd__(self,other):
        return self+other

    def __add__(self,other):
        if(type(other) in [Real,Integer,Rational]):
            other=Complex(RR(other),0)
        return Complex(self.re+other.re,self.im+other.im)

    def __mul__(self,other):
        if(type(other) in [MonovariatePolynom,Vector]):
            return other*self
        if(type(other) in [Integer,Rational,Real]):
            other=Complex(other,0)
        if(type(other)==Complex):
            a=self.re
            b=self.im
            c=other.re
            d=other.im
            # (a+ib)(c+id)
            #=ac+iad+ibc-bd
            #=ac-bd+i(ad+bc)
            return Complex(a*c-b*d,a*d+b*c)
        raise NotImplementedError(f"Multiplication between Complex and {type(other).__name__} is not implemented")
    def __eq__(self,other):
        if(type(other) in [Real,Integer,Rational]):
            other=Complex(RR(other),0)
        return self.re==other.re and self.im==other.im
