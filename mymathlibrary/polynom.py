import mymathlibrary as mml


class Polynom:
    def __init__(self,varnames: str,array,coefs):
        self.array=array
        self.coefs=coefs
        self.varnames=varnames
        self.reduce()
    def reduce(self):
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
        self.coefs=mml.Vector(*newCoefsB)
        self.array=mml.Matrix(*newArrayB)

    def __str__(self) -> str:
        if(len(self.coefs)==0):
            return "0"
        subStrings=[]
        for e in range(self.array.size[0]):
            subString=[]
            if(abs(self.coefs[e])!=1 or self.array[e]==mml.Vector(*[0]*len(self.varnames))):
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
        if(mml.type(other) in [mml.Integer,mml.Real,mml.Rational]):
            return Polynom(self.varnames,self.array,other*self.coefs)
        if(mml.type(other) in [mml.Matrix]):
            return other*self
        assert mml.type(other)==Polynom, f"Multiplication between Polynom and {mml.type(other).__name__} is not implemented"
        # distribute
        for e1 in range(len(self.array)):
            for e2 in range(len(other.array)):
                newPolynomCoefs.append(self.coefs[e1]*other.coefs[e2])
                newPolynomArray.append(self.array[e1]+other.array[e2])
        mat=mml.Matrix(*newPolynomArray)
        toReturn=Polynom(self.varnames,mat,mml.Vector(*newPolynomCoefs))
        return toReturn
    def __pow__(self,other):
        assert mml.type(other)==mml.Integer, "can only use integer exponent on polynoms"
        newPolynom=Polynom(self.varnames,mml.Matrix([0]*len(self.varnames)),mml.Vector(1))
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
        if(mml.type(other) in [mml.Integer,mml.Real,mml.Rational,mml.Complex]):
            other=Polynom(self.varnames,mml.Matrix([0]*len(self.varnames)),mml.Vector(other)) 

        assert mml.type(other)==Polynom, f"Addition between Polynom and {type(other).__name__} is no implemented"
        for e in range(len(self.coefs)):
            newPolynomCoefs.append(self.coefs[e])
            newPolynomArray.append(self.array[e])
        for e in range(len(other.coefs)):
            newPolynomCoefs.append(other.coefs[e])
            newPolynomArray.append(other.array[e])

        return Polynom(self.varnames,mml.Matrix(*newPolynomArray),mml.Vector(*newPolynomCoefs))

    def __eq__(self, other) -> bool:
        #return self.coefs==other.coefs and self.array==other.array
        return (self-other).coefs==mml.Vector()
    
    def __sub__(self,other):
        if(type(other)==mml.Integer):
            other=Polynom(self.varnames,mml.Matrix([0,0,0]),mml.Vector(1))
        return self+Polynom(self.varnames,other.array,-other.coefs)

    def __rsub__(self,other):
        # other-self
        return -self+other

    def __hash__(self):
        return len(self.coefs)+hash(self.coefs)+hash(self.array)
