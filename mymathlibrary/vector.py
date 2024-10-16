import mymathlibrary as mml


class Vector:
    def __init__(self, *value: list):
        self.list=list(value)
    def colinearTo(self,other) -> bool:
        coef=None
        if(len(self)!=len(other)): return False
        for i in range(len(other)):
            if((self[i]==0 or other[i]==0) and self[i]!=other[i]):
                return False
            if(coef==None):
                if(self[i]!=0):
                    coef=other[i]/self[i]
            elif(coef!=other[i]/self[i]):
                return False
        return True

    def __repr__(self):
        return str(self)

    def pop(self,index):
        self.list.pop(index)

    def __len__(self):
        return len(self.list)

    def isFreeFrom(self,*vectors) -> bool:
        matrixArray=[self.list]
        for v in vectors:
            matrixArray.append(v.list)
        mat=Matrix(*matrixArray)
        return mat.det()!=0

    def __getitem__(self,key):
        return self.list[key]
    def __setitem__(self, key, value):
            self.list[key] = value
    def isNull(self)->bool:
        for e in self:
            if(abs(e)>10**-10):
                return False
        return True
    def __call__(self, *args):
        newList=[]
        for e in self:
            if(callable(e)):
                newList.append(e(*args))
            else:
                newList.append(e)
        return Vector(*newList)
    
    def __sub__(self,other):
        return self+(-other)

    def __add__(self,other):
        assert mml.type(other)==Vector, f"Vector can only be added to Vector {(mml.type(other)).__name__} provided"
        assert len(self)==len(other), "Vectors must have the same size to add"
        newList=[]
        for i in range(len(self)):
            newList.append(self[i]+other[i])
        return Vector(*newList)

    def __str__(self) -> str:
        return "("+", ".join(map(str,self))+")"

    def __truediv__(self,other):
        return self*(1/other)


    def __mul__(self,other):
        newList=[]
        # assert type(other) in [Integer,float,Rational,Polynom], f"Multiplication between Vector and {type(other).__name__} is not implemented"
        for i in range(len(self)):
            newList.append(self[i]*other)
        return Vector(*newList)

    def dot(self,other):
        sum=0
        for i in range(len(self)):
            sum+=self[i]*other[i]
        return sum

    def __mod__(self,modulo):
        newList=[]
        for e in self.list:
            newList.append(e%modulo)
        return Vector(*newList)

    def __eq__(self,other) -> bool:
        return self.list==other.list
    def __neg__(self):
        return -1*self

    def __rmul__(self,other):
        return self*other
    def __abs__(self):
        return (sum([e**2 for e in self]))**(1/2)
    def __hash__(self):
        return hash(tuple(self.list))

    def __getattr__(self,key):
        if(key=="null"):
            return Vector(*([0]*len(self)))
