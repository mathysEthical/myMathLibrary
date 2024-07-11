from typing import Any, Self
import math
import time
import builtins
import gc
import random
floatClass=gc.get_referents(float.__dict__)[0]
intClass=gc.get_referents(int.__dict__)[0]
complexClass=gc.get_referents(complex.__dict__)[0]
intClass['Id']= property(lambda self: 1)
intClass['value']= property(lambda self: self)
floatClass['value']= property(lambda self: self)
complexClass['value']= property(lambda self: self)

def sqrt(value):
    return value**(1/2)

def type(object):
    if(object.__class__==int):
        return Integer
    if(object.__class__==float):
        return Real
    return object.__class__
class famousConstant:
    def __init__(self,varname,value,propertiesDict) -> None:
        self.value=value
        self.properties=propertiesDict
        self.varname=varname
    def __str__(self) -> builtins.str:
        return self.varname
pi=famousConstant("pi",3.1415926535897932384626433832795,{"sin": 0})

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
def modInverse(A, M):
    g = gcdExtended(A, M)
    assert g==1,f"Inverse of {A} modulo {M} doesn't exist"
    # m is added to handle negative x
    res = (x % M + M) % M
    return res

class clock:
    def __init__(self):
      self.startTime()

    def startTime(self):
        self.startedTime=time.time() 
 
    def endTime(self,debug=None):
        delta=time.time()-self.startedTime
        if(delta>1):
            print(f"{delta} secs")    
            if(debug):
                print(debug)

def sin(input):
    if(type(input) in [Rational,Integer,Real]):
        return math.sin(input+0.0)
    raise NotImplementedError(f"sin of {type(input).__name__} is not implemented")

class Integer:
    def __init__(self,value: int):
        if(type(value)==bool):
            value=int(value)
        
        self.value=int(str(math.floor(round(value,0))))
    def __str__(self) -> builtins.str:
        return f"{self.value}"
    def __repr__(self) -> builtins.str:
        return str(self)
    def __radd__(self,other):
        return self+other

    def __hash__(self):
        return hash(self.value)

    def __rfloordiv__(self,other):
        if(type(other)==Integer):
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
        return ZZ(-self.value)

    def __add__(self,other):
        if(type(other)==Integer):
            return ZZ(self.value+other.value)
        if(type(other)==Rational):
            return QQ(self.value*other.b+other.a,other.b)
        if(type(other) in [complex,Real]):
            return self.value+other
        if(type(other) in [Polynom,MonovariatePolynom]):
            return other+self
        raise ValueError(f"Addition not implemented between Integer and {type(other).__name__}")
    def __rmul__(self,other):
        return self*other
    def __mul__(self,other):
        if(type(other)==Integer):
            return ZZ(self.value*other.value)
        if(type(other)==Real):
            return other*self.value
        if(type(other)==Rational):
            return QQ(self.value*other.a,other.b)
        if(type(other) in [Matrix,Polynom,Vector,MonovariatePolynom]):
            return other*self
        if(type(other)==complex):
            return self.value*other
        raise NotImplementedError(f"Multiplication between Integer and {type(other).__name__} is not implemented")
    def __floordiv__(self,other: Self) -> Self:
        return ZZ(self.value//other)
    
    def __rsub__(self,other):
        return self+other*-1

    def __rtruediv__(self,other):
        # other/self
        if(type(other)==Integer):
            return QQ(other,self)
        else:
            return other/self
    def __abs__(self):
        return ZZ(abs(self.value))
    def __mod__(self,other):
        if(type(other)==Integer):
            return ZZ(self.value%other.value)
        else:
            return self.value%other
    
    
    def __sub__(self,other):
        return self+other*-1

    def __eq__(self,other):
        if(type(other)==Real):
            return self.value==other
        if(type(other)==Rational):
            return QQ(self.value,1)==other
        if(type(other)==Integer):
            return self.value==other.value
        ValueError(f"Equality between Integer and {type(other)} is not implemented")
    
    def __pow__(self,power):
        if(type(power)==Integer):
            return ZZ(self.value**power)
        if(type(power) in [Rational,Real]):
            return RR(self.value**power)
        raise NotImplementedError(f"Exponentiation between Integer and {type(power).__name__} is not implemented")
    def __lt__(self,other):
        if(type(other)==int):
            other=ZZ(other)
        return self.value<other.value

    def __ge__(self,other):
        return self>other or self==other
    def __le__(self,other):
        return self<other or self==other

    def __gt__(self,other):
        if(type(other)==int):
            other=ZZ(other)
        return self.value>other.value

    def __truediv__(self,other):
        if(type(other) in [Integer]):
            return QQ(self,other)
        if(type(other) in [Real,complex]):
            return self.value/other
        if(type(other) in [Rational]):
            return self*QQ(other.b,other.a)
        raise ValueError(f"Division between Integer and {type(other).__name__} is not implemented")

def gcd(a: Integer, b: Integer) -> Integer:
#   if(a==0 or b==0):return 1
  a=abs(a)
  b=abs(b)
  while ((a % b) > 0):
    R = a % b
    a = b
    b = R
  return b

class Rational:

    def __new__(cls,a: Integer,b:Integer):
        if(int(str(b.value))!=0 and abs(int(str(a.value))/int(str(b.value)))%1<10**(-10)):
            return ZZ(math.floor(float(a.value/b.value)))
        obj = object.__new__(cls)
        return obj

    def __hash__(self):
        return hash(str(self))

    def __init__(self,a: Integer,b: Integer):
        if(type(a)!=Integer or type(b)!=Integer):
            raise ValueError(f"numerator and denominator of Rational must be Integer, got: {type(a).__name__} and {type(b).__name__}") 
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
        return round(RR(self.a/self.b),digits)

    def reduce(self):
        if(self.b<0):
            self.a*=-1
            self.b*=-1
        reduceFactor=gcd(self.a,self.b)
        self.a//=reduceFactor
        self.b//=reduceFactor

    def __float__(self):
        return float(self.a)/float(self.b)

    def __str__(self) -> str:
        if(self.b==1):
            return f"{self.a}"
        return f"{self.a}/{self.b}"
    
    def __radd__(self,other: Self) -> Self:
        return self+other

    def __add__(self,other):
        if(type(other)==Integer):
            numerator=self.a+other*self.b
            denominator=self.b
            return QQ(numerator,denominator)
        if(type(other) in [Polynom,Complex,MonovariatePolynom]):
            return other+self
        if(type(other)==Real):
            return self.a/self.b+other
        if(type(other)!=Rational):
            raise NotImplementedError(f"Addition between Rational and {type(other).__name__} is not implemented")
        numerator=self.a*other.b+other.a*self.b
        denominator=self.b*other.b
        return QQ(numerator,denominator)

    def __rsub__(self,other) -> Self:
        return -(self-other)

    def __sub__(self,other) -> Self:
        if(type(other) in [Integer]):
            denominator=self.b
            numerator=self.a-other*self.b
            return QQ(numerator,denominator)
        if(type(other) in [Real]):
            return RR(self.a/self.b)-other
        denominator=self.b*other.b
        numerator=self.a*other.b-other.a*self.b
        return QQ(numerator,denominator)

    def __rmul__(self,other: Self) -> Self:
        return self*other

    def __pow__(self,power):
        if(type(power) in [Integer]):
            return QQ(self.a**power,self.b**power)
        if(type(power) in [Real]):
            return (self.a/self.b)**power
        raise NotImplementedError(f"Exponentiation of Rational with {type(power).__name__} is not implemented")
    def __mul__(self,other: Self) -> Self:
        if(type(other) in [Matrix,Vector]):
            return other*self
        if(type(other)==Integer):
            return QQ(self.a*other,self.b)
        if(type(other) in [Real,complex,Complex]):
            return self.a/self.b*other 
        if(type(other)==Polynom):
            return other*self
        assert type(other)==Rational, f"Multiplication between Rational and {type(other).__name__} is not implemented"  
        denominator=self.b*other.b
        numerator=self.a*other.a
        return QQ(numerator,denominator)

    def inverse(self) -> Self:
        return QQ(self.b,self.a)

    def __truediv__(self,other: Self) -> Self:
        if(type(other) in [Rational]):
            otherInverse=other.inverse()
            return self*otherInverse
        elif(type(other) in [Integer]):
            otherInverse=QQ(1,other)
            return self*otherInverse
        if(type(other) in [Real]):
            return self.a/self.b/other
        raise NotImplementedError(f"Division between Rational and {type(other).__name__} is not implemented")

    def __rtruediv__(self,other: Self) -> Self:
        return other*self.inverse()
    
    def __abs__(self):
        return QQ(abs(self.a),abs(self.b))

    def __gt__(self,other) -> bool:
        if(type(other) in [Rational]):
            return self.a*other.b>other.a*self.b
        if(type(other) in [Real]):
            return RR(self.a/self.b)>other
        if(type(other) in [Integer]):
            return self.a>other*self.b
        raise NotImplementedError(f"Operator > between Rational and {type(other).__name__} is not implemented")
    def __neg__(self):
        return QQ(-self.a,self.b)

    def __repr__(self):
        return str(self)

    def __lt__(self,other):
        # self < other
        if(type(other)==Integer):
            return (self-other).a<0
        if(type(other) in [Real]):
            return self.a/self.b<other
        return (self-other).a<0

    def __pos__(self):
        return self
    def __rmod__(self,other):
        # other % self
        if(type(other) in [Real]):
            return other%(self.a/self.b)
        if(type(other) in [Integer]):
            other=QQ(other,1)
            return other%self


    def __mod__(self,modulo):
        if(type(modulo) in [Real]):
            return (self.a/self.b)%modulo
        if(type(modulo)==Integer):
            return self-self.a//(self.b*modulo)*modulo
        if(type(modulo) in [Rational]):
            return self-(self//modulo)*modulo
        else:
            assert False, f"Modulo operator between Rational and {type(modulo).__name__} is not implemented"

    def __floordiv__(self,other):
        if(type(other) in [Rational]):
            return ZZ((self.a*other.b)//(other.a*self.b))
        return ZZ((self.a/self.b)//other)

    def __eq__(self,other: Self) -> bool:
        if(type(other)==Rational):
            return self.a==other.a and self.b==other.b
        if(type(other)==Integer):
            return self.b==1 and self.a==other
        if(type(other)==Real):
            return self.a/self.b==other

class Vector:
    def __init__(self, *value: list):
        self.list=list(value)
    def colinearTo(self,other:Self) -> bool:
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
    
    def __sub__(self,other: Self)->Self:
        return self+(-other)

    def __add__(self,other: Self) -> Self:
        assert type(other)==Vector, f"Vector can only be added to Vector {(type(other)).__name__} provided"
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

    def dot(self,other: Self):
        sum=0
        for i in range(len(self)):
            sum+=self[i]*other[i]
        return sum

    def __mod__(self,modulo):
        newList=[]
        for e in self.list:
            newList.append(e%modulo)
        return Vector(*newList)

    def __eq__(self,other: Self) -> bool:
        return self.list==other.list
    def __neg__(self):
        return -1*self

    def __rmul__(self,other):
        return self*other
    def __abs__(self):
        return (sum([e**2 for e in self]))**(1/2)
    def __hash__(self) -> Integer:
        return hash(tuple(self.list))

    def __getattr__(self,key):
        if(key=="null"):
            return Vector(*([0]*len(self)))

def reorder(theObject,order):
    return type(theObject)(*[theObject[order[i]] for i in range(len(order))])

class Matrix:
    def __init__(self,*array,**kwargs):
        isIdentity = kwargs.get('isIdentity', False)
        self.array=[Vector(*row) for row in array]
        if(len(array)==0):
            self.size=(0,0)
        else:
            self.size=(len(array),len(array[0]))

    
        if(isIdentity):self.isIdentity=True

    def inverse(self) -> Self:
        det=self.det()
        assert det!=0, "matrix inverse doesn't exists, detrerminant is 0"
        return 1/self.det()*self.adjugate().T

    def __repr__(self):
        return str(self)

    def removeEmptyRow(self) -> bool:
        """
        Remove empty rows, return False if no empty row is found, return True otherwise
        """
        toReturn=False
        newArray=[]
        for v in self:
            nullVect=Vector(*([0]*len(v)))
            if(v!=nullVect):
                newArray.append(v)
            else:
                # print("not free")
                toReturn=True
        newMat=Matrix(*newArray)
        self.__dict__.update(newMat.__dict__)
        return toReturn

    def switchColums(self,cA,cB):
        colA=self.T[cA]
        colB=self.T[cB]
        newArray=self.T.array.copy()
        newArray[cA]=colB
        newArray[cB]=colA
        return Matrix(*newArray).T

    def eigen(self):
        lambdaCoef=QQ["l"]
        valeursPropres=(self-lambdaCoef*self.Id).det().roots()
        toReturn={}
        for r in valeursPropres:
            vecteursPropres=(self-self.Id*r).solveRight(self[0].null)
            toReturn[r]=vecteursPropres
        return toReturn
    def diag(self):
        lambdaCoef=QQ["l"]
        valeursPropres=(self-lambdaCoef*self.Id).det().roots()
        PArray=[]
        #print(f"{valeursPropres=}")
        for r in valeursPropres:
            vecteursPropres=(self-self.Id*r).solveRight(self[0].null)
            for v in list(vecteursPropres.set):
                PArray.append(v)
        P=Matrix(*PArray).T
        # return
        Pinv=P.inverse()

        return {"P": P, "D": diag(*valeursPropres), "Pinv": Pinv}

    def gaussianElimination(self,*args):
        order=list(range(len(self[0])))
        if(len(args)>0):
            order=args[0]
        toReturn=self.array.copy()
        matToReturn=Matrix(*toReturn)
        if(matToReturn.removeEmptyRow()):
            return matToReturn.gaussianElimination(order)
        toReturn=matToReturn.array
        if(toReturn[0][0]==0):
            toReturn=toReturn[1:]+toReturn[0:1]
        # assert self.size[0]+1==self.size[1], "Gaussian elimination requires (n,n+1) matrix size"
        # inf triangle
        # print("mat: \n",matToReturn)
        for x in range(matToReturn.size[0]-1):
            reducingWith=toReturn[x]
            if(reducingWith[x]==0):
                matToReturn=matToReturn.switchColums(x,x+1)
                order[x],order[x+1]=order[x+1],order[x]
                return matToReturn.gaussianElimination(order)
            # print(f"{reducingWith=}")
            for y in range(x+1,matToReturn.size[0]):
                if(toReturn[y][x]!=0):
                    # print(f"{reducingWith*toReturn[y][x]/reducingWith[x]=}")
                    toReturn[y]-=reducingWith*toReturn[y][x]/reducingWith[x]
                    # print(f"{toReturn[y]=}")
                    matToReturn=Matrix(*toReturn)
                    # print("mat: \n",matToReturn)
                    if(matToReturn.removeEmptyRow()):
                        # print("system was not free")
                        return matToReturn.gaussianElimination(order)
                    toReturn=matToReturn.array
        # print("inf triangle ready: ",matToReturn,"end")
        # sup triangle
        for x in range(matToReturn.size[0]-1,-1,-1):
            # print(x)
            reducingWith=toReturn[x]
            if(reducingWith[x]==0):
                matToReturn=matToReturn.switchColums(x,x+1)
                order[x],order[x+1]=order[x+1],order[x]
                return matToReturn.gaussianElimination(order)
            for y in range(x-1,-1,-1):
                if(toReturn[y][x]!=0):
                    toReturn[y]-=reducingWith*toReturn[y][x]/reducingWith[x]
                    matToReturn=Matrix(*toReturn)
                    # print("mat",matToReturn)
                    if(matToReturn.removeEmptyRow()):
                        # print("system was not free")
                        return matToReturn.gaussianElimination(order)
                    toReturn=matToReturn.array
        for y in range(self.size[0]):
            toReturn[y]/=toReturn[y][y]

        return (Matrix(*toReturn),order)
    def __pow__(self,other):
        assert type(other)==Integer, f"Exponentiation between Matrix and {type(other).__name__} is not implemented"
        toReturn=self.Id
        #self ** 15 = self^1*self^2*self^4*self^8
        if(other<0):
            return self.inverse()**abs(other)
        P=self

        while(other>=1):
            if(other%2):
                toReturn*=P
            other//=2
            P*=P
        return toReturn

    def index(self,value):
        for i in range(len(self)):
            if(self[i]==value):
                return i
        return -1
    def pop(self,index):
        self.array.pop(index)
        self=Matrix(*self.array)
    def adjugate(self) -> Self:
        newArray=[]
        for i in range(self.size[0]):
            newList=[]
            for j in range(self.size[0]):
                subArray=[]
                for iB in range(self.size[0]):
                    if(iB!=i):
                        subList=[]
                        for jB in range(self.size[0]): 
                            if(jB!=j):
                                subList.append(self[iB][jB])
                        subArray.append(subList)
                newList.append(Matrix(*subArray).det()*(-1)**(i+j))
            newArray.append(newList)
        return Matrix(*newArray)

    def __mod__(self,modulo):
        newMatrixArray=[]
        for v in self.array:
            newMatrixArray.append(v%modulo)
        return Matrix(*newMatrixArray)

    def det(self) -> Integer:
        assert self.size[0]==self.size[1], "Matrices must be square to have a determinant"

        if(self.size[0]==1):
            return self[0][0]
        
        total=0
        for x in range(self.size[0]):
            newArray=[]
            for i in range(1,self.size[0]):
                newList=[]
                for j in range(self.size[1]):
                    if(j!=x): 
                        newList.append(self[i][j])
                newArray.append(newList)
            total+=self[0][x]*Matrix(*newArray).det()*(-1)**x
        return total


    def transpose(self):
        newArray=[]
        for i in range(self.size[1]):
            newList=[]
            for j in range(self.size[0]):
                newList.append(self[j][i])
            newArray.append(newList)
        return Matrix(*newArray)

    def __len__(self):
        return self.size[0]

    def __str__(self) -> str:
        return "\n".join(map(str,self.array))
    
    def identity(self):
        identityArray=[[ZZ(x==y) for x in range(len(self))] for y in range(len(self))]
        return Matrix(*identityArray,isIdentity=True) 

    def __truediv__(self,other: Self) -> Self:
        return self*other.inverse()

    def __getitem__(self,key):
        return self.array[key]

    def solveRight(self,other) -> Self:
        # S*X=O
        if(type(other)==Vector):
            other=Matrix(other).T
        assert type(other)==Matrix, f"can't use solveRight on {type(other).__name__}"
        assert self.size[0]==other.size[0], "number of rows must be equal between self and other to solveRight"
        solutionArray=[]
        for v in other.T:
            equationArray=[]
            for i in range(len(self)):
                equationArray.append([*self[i]]+[v[i]])
            equationMat=Matrix(*equationArray)
            # print("eq:",equationMat)
            eliminated,order=equationMat.gaussianElimination()
            # print("eliminated:",eliminated)
            # print("order",order)
            solutionVector=eliminated.T[-1]
            solutionArray.append(solutionVector)
            if(solutionVector==solutionVector.null and len(eliminated)+2<=len(eliminated[0])):
                # Vect
                # print(solutionVector)
                # return Span(*([-eliminated[i][eliminated.size[1]-2] for i in range(len(solutionVector))]+[0]*(len(eliminated[0])-len(eliminated)-2)),1)
                # print(*[-eliminated[i][eliminated.size[1]-2] for i in range(len(solutionVector))],1)
                return Span(reorder(Vector(*[-eliminated[i][eliminated.size[1]-2] for i in range(len(solutionVector))],1),order[:-1]))
                # return Span(*[-eliminated[i][eliminated.size[1]-2] for i in range(len(solutionVector))],1)

        return Matrix(*solutionArray).T
    def solveLeft(self,other: Self) -> Self:
        # X*S=O
        # X=O*inv(S)
        # 
        return other*self.inverse()

    def __add__(self,other: Self) -> Self:
        assert type(other)==Matrix, f"Matrices can only be added to matrices, {type(other).__name__} provided"
        assert self.size==other.size, "Matrices must have the same size"
        newArray=[]
        for i in range(self.size[0]):
            newList=[]
            for j in range(self.size[1]):
                newList.append(self[i][j]+other[i][j])
            newArray.append(Vector(*newList))
        return Matrix(*newArray)

    def __sub__(self,other: Self):
        return self+other*-1
    
    def __mul__(self,other) -> Self:
        newArray=[]
        if(type(other) in [Complex,Integer,Real,Rational,Polynom,MonovariatePolynom]):
            for row in self:
                newArray.append((row*other).list)
            return Matrix(*newArray)
        if(type(other) in [modularRingObject]):
            return other.ring(self*other.object)
        assert type(other)==Matrix,f"Multiplication between Matrix and {type(other).__name__} is not implemented"
        assert self.size[1]==other.size[0], f"Matrices {self.size} and {other.size} are not compatible for multiplication"
        for i in range(self.size[0]):
            newList=[]
            for j in range(other.size[1]):
                newList.append(self[i].dot(other.T[j]))
            newArray.append(newList)
        return Matrix(*newArray)
    def __rmul__(self,other: Self) -> Self:
        return self*other
    def __call__(self, *args: Any):
        newArray=[]
        for v in self:
            newArray.append(v(*args))
        return Matrix(*newArray)
    def __eq__(self,other: Self) -> bool:
        if(self.size!=other.size): 
            return False
        for i in range(self.size[0]):
            if(self[i]!=other[i]): 
                return False
        return True
    def __hash__(self):
        return hash(tuple(self.array))
    def __getattr__(self,key):
        if(key=="T"):
            return self.transpose()
        if(key=="Id"):
            return self.identity()
class numberSystem:
    def __init__(self,value):
        self.value=value

    def __str__(self) -> str:
        self.value.__name__
    
    def __call__(self, *args) -> Any:
        return self.value(*args)

    def __getitem__(self,key):
        varname="".join(list(key))
        if(len(key)==1):
            return MonovariatePolynom(varname,Vector(0,1))
        if(len(key)>1):
            return [Polynom(varname,Matrix([int(i==k) for k in range(len(key))]),Vector(1)) for i in range(len(key))]

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
        return self.ring(self.object*modInverse(other,self.ring.modulo))
    def __mod__(self,other):
        return self.ring(self.object%other)
    def __str__(self) -> str:
        return f"{self.object} (mod {self.ring.modulo})"
    def __eq__(self,other):
        if(type(other)!=modularRingObject):
            return False
        return self.object==other.object
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.ring(self.object(*args))
    
    def inverse(self):
        if(type(self.object)==Integer):
            return modInverse(self.object,self.ring.modulo)
    def __getattr__(self,key):
        return self.ring(getattr(self.object,key))

class modularRing:
    def __init__(self,modulo):
        self.modulo=modulo
    
    def __call__(self, value) -> Any:
        if(type(value)==modularRingObject and value.ring.modulo==self.modulo):
            return value
        return modularRingObject(self,value%self.modulo)
    def __contains__(self,other):
        if(type(other)==modularRingObject):
            other=other.object
        return self(other).object==other

class Real:

    def __new__(cls,value):
        obj = object.__new__(cls)
        if(type(value)==complex):
            return value
        if(abs(value-round(value,0))<10**(-6)):
            return ZZ(value)
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
        if(type(other) in [Real,Integer]):
            return RR(self.value+other.value)
        if(type(other) in [MonovariatePolynom]):
            return other+self
        raise NotImplementedError(f"Addition between Real and {type(other).__name__} is not implemented")
    def __float__(self):
        return float(self.value)

    def __rtruediv__(self,other):
        # other/self
        if(type(other) in [Real,Integer]):
            return RR(other.value*(1/self.value))

        raise NotImplementedError(f"Division between {type(other).__name__} and Real is not implemented")
    def __truediv__(self,other):
        if(type(other) in [Real,Integer]):
            return RR(self.value/other.value)
        if(type(other) in [Rational]):
            return RR(self.value*other.b/other.a)
        raise NotImplementedError(f"Division between Real and {type(other).__name__} is not implemented")
    def __neg__(self):
        return RR(-self.value)

    def __rmul__(self,other):
        return RR(self*other)

    def __round__(self,digits):
        return round(self.value,digits)

    def __lt__(self,other) -> bool:
        return self.value<other.value

    def __mul__(self,other):
        if(type(other) in [Rational,Vector]):
            return other*self
        if(type(other) in [Real,Integer]):
            return RR(self.value*other.value)
        if(type(other) in [Matrix]):
            return other*self
        raise NotImplementedError(f"Multiplication between Real and {type(other).__name__} is not implemented")
    def __rsub__(self,other):
        #other-self
        return RR(other.value-self.value)
    def __sub__(self,other):
        return RR(self.value-other.value)

    def __eq__(self,other):
        return self.value==other

    def __abs__(self):
        return abs(self.value)

    def __pow__(self,power):
        return RR(self.value)**float(power)
    def __hash__(self):
        return hash(self.value)
    def __gt__(self,other):
        if(type(other) in [Real]):
            return self.value>other.value
        raise NotImplementedError(f"Operator > between Real and {type(other).__name__} is not implemented")
class Point:
    def __init__(self, *coords: list) -> None:
        self.coords=coords
    def __eq__(self, other: Self) -> bool:
        return self.coords==other.coords

def distance(pointA: Point,pointB: Point)-> Real:
    return sum((pointA.coords[i]-pointB.coords[i])**2 for i in range(len(pointA.coords)))**0.5    

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
ZZ=numberSystem(Integer)
QQ=numberSystem(Rational)
RR=numberSystem(Real)
CC=numberSystem(Complex)

i=CC(0,1)

def degreeThreeRoots(d,c,b,a):
    roots=[]
    epsilon=(-1+sqrt(-3))/2
    delta0=b**2-3*a*c
    delta1=2*b**3-9*a*b*c+27*a**2*d
    C=((delta1+sqrt(delta1**2-4*delta0**3))/2)**(1/3)
    for k in range(3):
        frac=ZZ(0)
        if(C!=0 and delta0!=0):
            frac=delta0/(epsilon**k*C)
        xk=-1/(3*a)*(b+epsilon**k*C+frac)
        roots.append(xk)
    # C=((delta1-sqrt(delta1**2-4*delta0**3))/2)**(1/3)
    # for k in range(3):
    #     frac=Integer(0)
    #     if(C!=0 and delta0!=0):
    #         frac=delta0/(epsilon**k*C)
    #     xk=-1/(3*a)*(b+epsilon**k*C+frac)
    #     roots.append(xk)
    return roots
def degreeTwoRoots(c,b,a):
    delta=b**2-4*a*c
    r1=(-b+sqrt(delta))/(2*a)
    r2=(-b-sqrt(delta))/(2*a)
    return [r1,r2]

class MonovariatePolynom:
    def __init__(self,varname: str,coefs: Vector) -> None:
        self.coefs=coefs
        self.varname=varname
        self.reduce()
        self.degree=len(self.coefs)-1

    def reduce(self):
        while(len(self.coefs)>0 and self.coefs[-1]==0):
            self.coefs.pop(-1)

    def __sub__(self,other):
        return -other+self

    def __call__(self, input):
        return sum([self.coefs[i]*input**i for i in range(len(self.coefs))])

    def __neg__(self):
        return MonovariatePolynom(self.varname,-self.coefs) 

    def __radd__(self,other):
        return self+other

    def derivate(self):
        return MonovariatePolynom(self.varname,Vector(*[self.coefs[i+1]*(i+1) for i in range(len(self.coefs)-1)]))

    def __floordiv__(self,other):
        return euclidianPolynomialDivision(self,other)
    def findOneRoot(self):
        if(self.degree<=3):
            if(self.degree==3):
                roots=degreeThreeRoots(*self.coefs)
                for r in roots:
                    if(r.imag==0):
                        return RR(r.real)
            if(self.degree==2):
                roots=degreeTwoRoots(*self.coefs)
                return roots[0]
            if(self.degree==1):
                return -self.coefs[0]/self.coefs[1]
        actualGuess=random.random()*100-50
        delta=1
        iteration=0
        while(delta>10**(-400)):
            iteration+=1
            a=self(actualGuess)
            b=self.derivate()(actualGuess)
            if(a==0): break
            
            if(b!=0):
                delta=actualGuess
                actualGuess=actualGuess-a/b
                delta=abs(actualGuess-delta)
            else:
                actualGuess=random.random()*100-50
            if(iteration>10**3):
                iteration=0
                # print("reset iterations on the poly:",self)
                actualGuess=random.random()*100-50
        # print("beforeRound:",actualGuess)
        return continuedFractions(round(actualGuess,16))
    
    def factor(self):
        roots=self.roots()
        factorsList=[]
        product=1
        for r in roots:
            factorsList.append(self.var-r)
            product*=self.var-r
        factorsList.append(self//(product))
        return factorsList
    
    def roots(self):
        actualPoly=self
        rootsList=[]
        if(self.degree<=3):
            if(self.degree==3):
                roots=degreeThreeRoots(*self.coefs)
                for r in roots:
                    if(RR(r.imag)==0):
                        rootsList.append(RR(r.real))
                return rootsList
            if(self.degree==2):
                roots=degreeTwoRoots(*self.coefs)
                return roots
            if(self.degree==1):
                return -self.coefs[0]/self.coefs[1]
        while(actualPoly.degree>0):
            foundRoot=actualPoly.findOneRoot()
            rootsList.append(foundRoot)
            actualPoly=euclidianPolynomialDivision(actualPoly,self.var-foundRoot)
        return rootsList

    def __add__(self,other: Self):
        if(type(other) in [Integer,Rational,Real,Complex]):
            other=self.Id*other
        biggest=max(len(self.coefs),len(other.coefs))
        thisCoefs=Vector(*(self.coefs.list+[0]*(biggest-len(self.coefs))))
        otherCoefs=Vector(*(other.coefs.list+[0]*(biggest-len(other.coefs))))
        newCoefs=thisCoefs+otherCoefs        
        return MonovariatePolynom(self.varname,newCoefs)

    def __pow__(self,other):
        assert type(other)==Integer, "can only use integer exponent on polynoms"
        newPolynom=MonovariatePolynom(self.varname,Vector(1))
        if(other<0):
            raise NotImplementedError("Negative exponentiation of polynoms not implemented")
        for i in range(other):
            newPolynom*=self
        return newPolynom

    def __eq__(self,other):
        return (self-other).coefs.isNull()

    def __rmul__(self,other):
        return self*other

    def __mul__(self,other):
        if(type(other) in [Integer,Rational,Real,Complex]):
            return MonovariatePolynom(self.varname,other*self.coefs) 
        if(type(other) in [Matrix]):
            return other*self
        assert type(other)==MonovariatePolynom, f"Multiplication between MonovariatePolynom and {type(other).__name__} is not implemented"
        if(type(other)==MonovariatePolynom):
            newPolynomCoefs=[0]*len(self.coefs)*len(other.coefs)
            # distribute
            for e1 in range(len(self.coefs)):
                for e2 in range(len(other.coefs)):
                    newPolynomCoefs[e1+e2]+=(self.coefs[e1]*other.coefs[e2])
            toReturn=MonovariatePolynom(self.varname,Vector(*newPolynomCoefs))
            return toReturn
    def __str__(self) -> builtins.str:
        if(len(self.coefs)==0):
            return "0"
        subStrings=[]
        for e in reversed(range(len(self.coefs))):
            subString=[]
            if(self.coefs[e]!=0):
                if((self.coefs[e]!=1 and self.coefs[e]!=-1) or e==0):
                    subString.append(f"{self.coefs[e]}")
                c=self.varname
                if(self.coefs[e]==-1):
                    c="-"+c
                if(e!=0):
                    if(e==1):
                        subString.append(f"{c}")
                    else:
                        subString.append(f"{c}**{e}")
                subStrings.append("*".join(subString))
        finalStr=subStrings[0]
        for s in range(1,len(subStrings)):
            subS=subStrings[s]
            if(subS.startswith("-")):
                finalStr+=subS
            else:
                finalStr+="+"+subS
        return finalStr
    
    def __repr__(self) -> builtins.str:
        return str(self)
    def __getattr__(self,key):
        if(key=="Id"):
            return MonovariatePolynom(self.varname,Vector(1))
        if(key=="var"):
            return MonovariatePolynom(self.varname,Vector(0,1))
class Polynom:
    def __init__(self,varnames: str,array: Matrix,coefs: Vector):
        self.array=array
        self.coefs=coefs
        self.varnames=varnames
        self.reduce()
    def reduce(self) -> Self:
        newCoefs=[]
        newArray=[]
        for i in range(len(self.coefs)):
            if(self.array[i] in newArray):
                foundIndex=newArray.index(self.array[i])
                newCoefs[foundIndex]+=self.coefs[i]
            else:
                newCoefs.append(self.coefs[i])
                newArray.append(self.array[i])
        
        newCoefsB=[]
        newArrayB=[]
        for i in range(len(newCoefs)):
            if(newCoefs[i]!=0):
                newArrayB.append(newArray[i])
                newCoefsB.append(newCoefs[i])
        self.coefs=Vector(*newCoefsB)
        self.array=Matrix(*newArrayB)

    def __str__(self) -> str:
        if(len(self.coefs)==0):
            return "0"
        subStrings=[]
        for e in range(self.array.size[0]):
            subString=[]
            if(abs(self.coefs[e])!=1 or self.array[e]==Vector(*[0]*len(self.varnames))):
                subString.append(f"{self.coefs[e]}")
            for c in self.varnames:
                exp=self.array[e][self.varnames.index(c)]
                if(self.coefs[e]==-1):
                    c="-"+c
                if(exp!=0):
                    if(exp==1):
                        subString.append(f"{c}")
                    else:
                        subString.append(f"{c}**{exp}")
            subStrings.append("*".join(subString))
        finalStr=subStrings[0]
        for s in range(1,len(subStrings)):
            subS=subStrings[s]
            if(subS.startswith("-")):
                finalStr+=subS
            else:
                finalStr+="+"+subS
        return finalStr
    def __call__(self, *args):
        assert len(args)==len(self.varnames), f"Number of arguments ({len(args)}) doesn't match number of variables ({len(self.varnames)})"
        total=0
        for i in range(len(self.coefs)):
            product=self.coefs[i]
            for v in range(len(self.varnames)):
                product*=args[v]**self.array[i][v]
            total+=product
        return total

    def __rmul__(self,other):
        return self*other
    def __mul__(self,other):
        newPolynomArray=[]
        newPolynomCoefs=[]
        if(type(other) in [Integer,Real,Rational]):
            return Polynom(self.varnames,self.array,other*self.coefs)
        if(type(other) in [Matrix]):
            return other*self
        assert type(other)==Polynom, f"Multiplication between Polynom and {type(other).__name__} is not implemented"
        # distribute
        for e1 in range(len(self.array)):
            for e2 in range(len(other.array)):
                newPolynomCoefs.append(self.coefs[e1]*other.coefs[e2])
                newPolynomArray.append(self.array[e1]+other.array[e2])
        mat=Matrix(*newPolynomArray)
        toReturn=Polynom(self.varnames,mat,Vector(*newPolynomCoefs))
        return toReturn
    def __pow__(self,other):
        assert type(other)==Integer, "can only use integer exponent on polynoms"
        newPolynom=Polynom(self.varnames,Matrix([0]*len(self.varnames)),Vector(1))
        assert other>=0, "Negative exponentiation of polynoms not implemented"
        for i in range(other):
            newPolynom*=self
        return newPolynom
    
    def __neg__(self):
        return -1*self

    def __radd__(self,other):
        return self+other

    def __mod__(self,modulo):
        return self

    def __add__(self,other):
        newPolynomArray=[]
        newPolynomCoefs=[]
        if(type(other) in [Integer,Real,Rational,Complex]):
            other=Polynom(self.varnames,Matrix([0]*len(self.varnames)),Vector(other)) 

        assert type(other)==Polynom, f"Addition between Polynom and {type(other).__name__} is no implemented"
        for e in range(len(self.coefs)):
            newPolynomCoefs.append(self.coefs[e])
            newPolynomArray.append(self.array[e])
        for e in range(len(other.coefs)):
            newPolynomCoefs.append(other.coefs[e])
            newPolynomArray.append(other.array[e])

        return Polynom(self.varnames,Matrix(*newPolynomArray),Vector(*newPolynomCoefs))

    def __eq__(self, other) -> bool:
        #return self.coefs==other.coefs and self.array==other.array
        return (self-other).coefs==Vector()
    
    def __sub__(self,other):
        if(type(other)==Integer):
            other=Polynom(self.varnames,Matrix([0,0,0]),Vector(1))
        return self+Polynom(self.varnames,other.array,-other.coefs)

    def __rsub__(self,other):
        # other-self
        return -self+other

    def __hash__(self) -> Integer:
        return len(self.coefs)+hash(self.coefs)+hash(self.array)

def continuedFractions(input: Real) -> Rational:
    if(type(input)==Integer):return input
    resultIntegers=[]
    if(input<0):
        return -continuedFractions(-input)
    actual=input
    for _ in range(100):
        integerPart=math.floor(float(actual))
        resultIntegers.append(integerPart)
        actual-=integerPart
        if(actual==0): break
        actual=1/actual
    ratio=QQ(0,1)
    for b in reversed(resultIntegers):
        ratio+=b
        if(ratio==0): break
        ratio=1/ratio
    if(ratio==0): return ratio
    return 1/ratio

class Product:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def __str__(self):
        return f"{self.a}*{self.b}"

class symbolicExpression:
    def __init__(self,expressionArray):
        self.expressionArray=expressionArray
        self.expressionArray.sort(key=lambda element: type(element) in [str,Product])
        self.reduce()
    def reduce(self) -> None:
        if(len(self.expressionArray)<2): return
        while(type(self.expressionArray[1]) not in [str,Product]):
            self.expressionArray=[self.expressionArray[0]+self.expressionArray[1]]+self.expressionArray[2:]

    def __rmul__(self,other):
        return symbolicExpression([Product(other,self)])

    def __mul__(self,other):
        return symbolicExpression([Product(self,other)])

    def __add__(self,other):
        return symbolicExpression(self.expressionArray+[other]) 
    def __radd__(self,other):
        return symbolicExpression(self.expressionArray+[other]) 

    def __str__(self):
        return "+".join([str(b) for b in self.expressionArray])

def findFirstDiff(listA,listB):
    for i in range(len(listA)):
        if(len(listA)==i or len(listB)==i):
            return i
        if(round(listA[i],15)!=round(listB[i],15)):
            return i
    return -1
def euclidianPolynomialDivision(poly: MonovariatePolynom,factor: MonovariatePolynom) -> MonovariatePolynom:
    if(poly==factor):
        return poly.Id
    naturalGoal=poly.coefs.list[::-1]
    actualPolyArray=[0]*(poly.degree+1)
    actualMul=MonovariatePolynom(poly.varname,Vector(*actualPolyArray))*factor
    while(actualMul!=poly):
        extendedMul=(actualMul.coefs.list+[0]*(len(actualPolyArray)-len(actualMul.coefs)))[::-1]
        differenceIndex=findFirstDiff(naturalGoal,extendedMul)
        # print(f"difference between {naturalGoal} and {extendedMul} is at index ",differenceIndex)
        delta=naturalGoal[differenceIndex]-extendedMul[differenceIndex]
        iGood=(len(actualPolyArray)-1)-differenceIndex
        indexToEdit=iGood-factor.degree
        # print("indexToEdit: ",indexToEdit)
        actualPolyArray[indexToEdit]+=delta/factor.coefs[-1]
        actualMul=MonovariatePolynom(poly.varname,Vector(*actualPolyArray))*factor
    return MonovariatePolynom(poly.varname,Vector(*actualPolyArray))

class Span:
    def __init__(self,*vectors: Vector):
        self.set=set(vectors)
        self.reduce()

    def __repr__(self) -> builtins.str:
        return str(self)

    def reduce(self):
        # print("toReduce:",self.vect)
        newSetArray=[]
        for vect in self.set:
            newElementsArray=[]
            for e in vect:
                newElementsArray.append(continuedFractions(RR(e)))
            # print(f"{newElementsArray=}")
            vect=Vector(*newElementsArray)
            multiplier=1
            negative=0
            for e in vect:
                if(type(e)==Rational):
                    multiplier*=e.b
                if(e<0):
                    negative+=1
            vect*=multiplier
            
            if(negative*2>len(vect)):
                vect*=-1
            
            actualGCD=None
            for e in vect:
                if(e!=0):
                    if(actualGCD==None): actualGCD=e
                    else:actualGCD=gcd(actualGCD,e)
            if(actualGCD!=None):
                vect/=actualGCD
            newSetArray.append(vect)
        self.set=set(newSetArray)
    def __str__(self) -> builtins.str:
        return f"Span({self.set})"

def diag(*diagList):
    return Matrix(*[[0]*i+[diagList[i]]+[0]*(len(diagList)-i-1) for i in range(len(diagList))])

def var(varname):
    return symbolicExpression([varname])

def crt(remainders: list,modulus: list):
    prod=1
    for m in modulus: prod*=m
    finalRing=modularRing(prod)
    toReturn=finalRing(0)
    for r,m in zip(remainders,modulus):
        ProdI=prod//m
        localRing=modularRing(m)
        inv=localRing(ProdI)**-1
        toReturn+=r*inv.object*ProdI
    return toReturn

print(crt([1,4,6],[3,5,7]))