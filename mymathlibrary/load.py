from typing import Any, Self
import math
import time
import builtins
import gc
import random
from mymathlibrary.gcd import *
from mymathlibrary.vector import *
from mymathlibrary.real import *
from mymathlibrary.rational import *
from mymathlibrary.integer import *
from mymathlibrary.complex import *
from mymathlibrary.clock import *
from mymathlibrary.gcdextended import *
from mymathlibrary.span import *
from mymathlibrary.euclidianPolynomialDivision import *
from mymathlibrary.polynom import *
from mymathlibrary.modularRingObject import *

class numberSystem:
    def __init__(self,value):
        self.value=value

    def __str__(self) -> str:
        self.value.__name__
    
    def __call__(self, *args):
        return self.value(*args)

    def __getitem__(self,key):
        varname="".join(list(key))
        if(len(key)==1):
            return MonovariatePolynom(varname,Vector(0,1))
        if(len(key)>1):
            return [Polynom(varname,Matrix([int(i==k) for k in range(len(key))]),Vector(1)) for i in range(len(key))]



ZZ=numberSystem(Integer)
QQ=numberSystem(Rational)
RR=numberSystem(Real)
CC=numberSystem(Complex)


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
        print(f"Solving: {self}*X={other}")
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
            eliminated,order=equationMat.gaussianElimination()

            solutionVector=eliminated.T[-1]
            solutionArray.append(solutionVector)
            if(solutionVector==solutionVector.null and len(eliminated)+2<=len(eliminated[0])):
                print(f"OHH: {eliminated}")
                return Span(reorder(Vector(*[-eliminated[i][eliminated.size[1]-2] for i in range(len(solutionVector))],1),order[:-1]))

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

from mymathlibrary.monovariatePolynom import *


floatClass=gc.get_referents(float.__dict__)[0]
intClass=gc.get_referents(int.__dict__)[0]
complexClass=gc.get_referents(complex.__dict__)[0]
intClass['Id']= property(lambda self: 1)
intClass['value']= property(lambda self: self)
floatClass['value']= property(lambda self: self)

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



def modInverse(x, modulus):
    return pow(x,-1,modulus)


def sin(input):
    if(type(input) in [Rational,Integer,Real]):
        return math.sin(input+0.0)
    raise NotImplementedError(f"sin of {type(input).__name__} is not implemented")



def reorder(theObject,order):
    return type(theObject)(*[theObject[order[i]] for i in range(len(order))])


class modularRing:
    def __init__(self,modulo):
        self.modulo=modulo
    
    def __call__(self, value) -> Any:
        if(mml.type(value)==modularRingObject and value.ring.modulo==self.modulo):
            return value
        return modularRingObject(self,value%self.modulo)
    def __contains__(self,other):
        if(type(other)==modularRingObject):
            other=other.object
        return self(other).object==other

class Point:
    def __init__(self, *coords: list) -> None:
        self.coords=coords
    def __eq__(self, other: Self) -> bool:
        return self.coords==other.coords

def distance(pointA: Point,pointB: Point)-> Real:
    return sum((pointA.coords[i]-pointB.coords[i])**2 for i in range(len(pointA.coords)))**0.5    


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

# def euclidianPolynomialDivision(poly: MonovariatePolynom,factor: MonovariatePolynom) -> MonovariatePolynom:
#     if(poly==factor):
#         return poly.Id
#     naturalGoal=poly.coefs.list[::-1]
#     actualPolyArray=[0]*(poly.degree+1)
#     actualMul=MonovariatePolynom(poly.varname,Vector(*actualPolyArray))*factor
#     while(actualMul!=poly):
#         extendedMul=(actualMul.coefs.list+[0]*(len(actualPolyArray)-len(actualMul.coefs)))[::-1]
#         differenceIndex=findFirstDiff(naturalGoal,extendedMul)
#         # print(f"difference between {naturalGoal} and {extendedMul} is at index ",differenceIndex)
#         delta=naturalGoal[differenceIndex]-extendedMul[differenceIndex]
#         iGood=(len(actualPolyArray)-1)-differenceIndex
#         indexToEdit=iGood-factor.degree
#         # print("indexToEdit: ",indexToEdit)
#         actualPolyArray[indexToEdit]+=delta/factor.coefs[-1]
#         actualMul=MonovariatePolynom(poly.varname,Vector(*actualPolyArray))*factor
#     return MonovariatePolynom(poly.varname,Vector(*actualPolyArray))


def diag(*diagList):
    return Matrix(*[[0]*i+[diagList[i]]+[0]*(len(diagList)-i-1) for i in range(len(diagList))])

def var(varname):
    return symbolicExpression([varname])

def crt (remainders: list,modulus: list):

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
