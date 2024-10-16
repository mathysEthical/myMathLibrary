import mymathlibrary as mml

class modularRingObject:
    def __init__(self,ring,object):
        self.ring=ring
        self.object=object
    def __add__(self,other):
        return self.ring(self.object+other)
    def __radd__(self,other):
        return self.ring(self.object+other)
    def __sub__(self,other):
        return self.ring(self.object-other)
    def __mul__(self,other):
        return self.ring(self.object*other)
    def __rmul__(self,other):
        return self.ring(self.object*other)
    def __pow__(self,other):
        toReturn=self.Id.object
        #self ** 15 = self^1*self^2*self^4*self^8
        if(other<0):
            return self.ring(self.inverse())**abs(other)
        P=self

        while(other>=1):
            if(other%2):
                toReturn*=P
            other//=2
            P*=P
        # return toReturn
        return toReturn
    def __truediv__(self,other):
        return self.ring(self.object*mml.modInverse(other,self.ring.modulo))
    def __mod__(self,other):
        return self.ring(self.object%other)
    def __str__(self) -> str:
        return f"{self.object} (mod {self.ring.modulo})"
    def __eq__(self,other):
        if(type(other)!=modularRingObject):
            return False
        return self.object==other.object
    def __call__(self, *args, **kwds):
        return self.ring(self.object(*args))
    
    def inverse(self):
        if(mml.type(self.object)==mml.Integer):
            return mml.modInverse(self.object,self.ring.modulo)
    def __getattr__(self,key):
        return self.ring(getattr(self.object,key))
