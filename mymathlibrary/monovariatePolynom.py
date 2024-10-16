import mymathlibrary as mml
import random

class MonovariatePolynom:
    def __init__(self,varname: str,coefs):
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
        return MonovariatePolynom(self.varname,mml.Vector(*[self.coefs[i+1]*(i+1) for i in range(len(self.coefs)-1)]))

    def __floordiv__(self,other):
        return mml.euclidianPolynomialDivision(self,other)
    def findOneRoot(self):
        if(self.degree<=3):
            if(self.degree==3):
                roots=mml.degreeThreeRoots(*self.coefs)
                for r in roots:
                    if(r.imag==0):
                        return mml.RR(r.real)
            if(self.degree==2):
                roots=mml.degreeTwoRoots(*self.coefs)
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
        return mml.continuedFractions(round(actualGuess,16))
    
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
                roots=mml.degreeThreeRoots(*self.coefs)
                for r in roots:
                    if(mml.RR(r.imag)==0):
                        rootsList.append(mml.RR(r.real))
                return rootsList
            if(self.degree==2):
                roots=mml.degreeTwoRoots(*self.coefs)
                return roots
            if(self.degree==1):
                return [-self.coefs[0]/self.coefs[1]]
        while(actualPoly.degree>0):
            foundRoot=actualPoly.findOneRoot()
            rootsList.append(foundRoot)
            actualPoly=mml.euclidianPolynomialDivision(actualPoly,self.var-foundRoot)
        return rootsList

    def __add__(self,other):
        if(mml.type(other) in [mml.Integer,mml.Rational,mml.Real,mml.Complex]):
            other=self.Id*other
        biggest=max(len(self.coefs),len(other.coefs))
        thisCoefs=mml.Vector(*(self.coefs.list+[0]*(biggest-len(self.coefs))))
        otherCoefs=mml.Vector(*(other.coefs.list+[0]*(biggest-len(other.coefs))))
        newCoefs=thisCoefs+otherCoefs        
        return MonovariatePolynom(self.varname,newCoefs)

    def __pow__(self,other):
        assert mml.type(other)==mml.Integer, "can only use integer exponent on polynoms"
        newPolynom=MonovariatePolynom(self.varname,mml.Vector(1))
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
        if(mml.type(other) in [mml.Integer,mml.Rational,mml.Real,mml.Complex]):
            return MonovariatePolynom(self.varname,other*self.coefs) 
        if(mml.type(other) in [mml.Matrix]):
            return other*self
        assert mml.type(other)==MonovariatePolynom, f"Multiplication between MonovariatePolynom and {mml.type(other).__name__} is not implemented"
        if(mml.type(other)==MonovariatePolynom):
            newPolynomCoefs=[0]*len(self.coefs)*len(other.coefs)
            # distribute
            for e1 in range(len(self.coefs)):
                for e2 in range(len(other.coefs)):
                    newPolynomCoefs[e1+e2]+=(self.coefs[e1]*other.coefs[e2])
            toReturn=MonovariatePolynom(self.varname,mml.Vector(*newPolynomCoefs))
            return toReturn
    def __str__(self):
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
    
    def __repr__(self):
        return str(self)
    def __getattr__(self,key):
        if(key=="Id"):
            return MonovariatePolynom(self.varname,mml.Vector(1))
        if(key=="var"):
            return MonovariatePolynom(self.varname,mml.Vector(0,1))
