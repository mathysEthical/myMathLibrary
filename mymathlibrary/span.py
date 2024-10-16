import mymathlibrary as mml


class Span:
    def __init__(self,*vectors):
        self.set=set(vectors)
        self.reduce()

    def __repr__(self):
        return str(self)

    def reduce(self):
        # print("toReduce:",self.vect)
        newSetArray=[]
        for vect in self.set:
            newElementsArray=[]
            for e in vect:
                newElementsArray.append(mml.continuedFractions(mml.RR(e)))
            # print(f"{newElementsArray=}")
            vect=mml.Vector(*newElementsArray)
            multiplier=1
            negative=0
            for e in vect:
                if(type(e)==mml.Rational):
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
                    else:actualGCD=mml.gcd(actualGCD,e)
            if(actualGCD!=None):
                vect/=actualGCD
            newSetArray.append(vect)
        self.set=set(newSetArray)
    def __str__(self):
        return f"Span({self.set})"
