import mymathlibrary as mml

def euclidianPolynomialDivision(poly,factor) :
    if(poly==factor):
        return poly.Id
    naturalGoal=poly.coefs.list[::-1]
    actualPolyArray=[0]*(poly.degree+1)
    actualMul=mml.MonovariatePolynom(poly.varname,mml.Vector(*actualPolyArray))*factor
    while(actualMul!=poly):
        extendedMul=(actualMul.coefs.list+[0]*(len(actualPolyArray)-len(actualMul.coefs)))[::-1]
        differenceIndex=mml.findFirstDiff(naturalGoal,extendedMul)
        # print(f"difference between {naturalGoal} and {extendedMul} is at index ",differenceIndex)
        delta=naturalGoal[differenceIndex]-extendedMul[differenceIndex]
        iGood=(len(actualPolyArray)-1)-differenceIndex
        indexToEdit=iGood-factor.degree
        # print("indexToEdit: ",indexToEdit)
        actualPolyArray[indexToEdit]+=delta/factor.coefs[-1]
        actualMul=mml.MonovariatePolynom(poly.varname,mml.Vector(*actualPolyArray))*factor
    return mml.MonovariatePolynom(poly.varname,mml.Vector(*actualPolyArray))
