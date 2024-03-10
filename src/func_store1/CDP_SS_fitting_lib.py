import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize 
import pandas as pd
import os

def strainStress(peakStress,peakStrain,modulus):
    numPoint = 1000
    strainMin = 0
    strainMax = 0.01
    strainRan = strainMax - strainMin
    alpha = 0.157*pow(peakStress,0.785) - 0.905
    strain = np.array([x*strainRan/numPoint for x in range(numPoint)])
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    return strain, stress

def CDPStress(peakStress, peakStrain, modulus, strain):
    numPoint = len(strain)
    alpha = 0.157*pow(peakStress,0.785) - 0.905
    #strain = np.array([x*strainRan/numPoint for x in range(numPoint)])
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    return strain, stress

def CDPCost(peakStress, peakStrain, modulus, expStrain, expStress):
    strain = np.copy(expStrain)
    numPoint = len(strain)
    alpha = 0.157*pow(peakStress,0.785) - 0.905
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    cost = 0.5*np.sum(np.power((stress - expStress), 2))
    return cost

def CDPStressAlphaScale(peakStress, peakStrain, modulus, alphaScale, strain):
    numPoint = len(strain)
    #alpha = (0.157*pow(peakStress,0.785) - 0.905) * alphaScale
    alpha = alphaScale
    #strain = np.array([x*strainRan/numPoint for x in range(numPoint)])
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    return strain, stress

def CDPCostAlphaScale(peakStress, peakStrain, modulus, alphaScale, expStrain, expStress):
    strain = np.copy(expStrain)
    numPoint = len(strain)
    #alpha = (0.157*pow(peakStress,0.785) - 0.905) * alphaScale
    alpha = alphaScale
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    cost = 0.5*np.sum(np.power((stress - expStress), 2))
    return cost

def CDPCost2AlphaScale(params, expStrain, expStress):
    peakStress, peakStrain, modulus, alphaScale = params
    strain = np.copy(expStrain)
    numPoint = len(strain)
    #alpha = (0.157*pow(peakStress,0.785) - 0.905) * alphaScale
    alpha = alphaScale
    arrayX = strain / peakStrain
    elasticPeakStress = peakStrain*modulus
    varRho = peakStress / elasticPeakStress
    varN = elasticPeakStress / ( elasticPeakStress - peakStress )
    damage = np.array([0.0 for x in range(numPoint)])
    for i in range(numPoint):
        tempX = arrayX[i]
        if(tempX < 1 or tempX==1):
            damage[i] = 1 - (varRho*varN)/(varN - 1 + pow(tempX,varN))
        if(tempX > 1):
            damage[i] = 1 - (varRho) / (alpha*pow(tempX-1,2) + tempX)
    stress = modulus*(1 - damage)*strain
    cost = 0.5*np.sum(np.power((stress - expStress), 2))
    return cost

def f_Ec_E_fromT(f20, Ec20, E20, temperature):
    fRatio = 1.008 + temperature / (450*math.log(temperature/5800.0))
    EcRatio = 1.0
    if(temperature > 200):
        EcRatio = (-0.1*f20 + 7.7)*(math.exp(0.01*temperature-5.8)/(1+math.exp(0.01*temperature-5.8)) - 0.0219) + 1.0
    ERatio = 1.033 - 0.00165*temperature
    fT = f20 * fRatio
    EcT = Ec20 * EcRatio
    ET = E20 * ERatio
    return fT, EcT, ET

def fit_CDP_alpha(initial_params, expStrain, expStress):
    def objective(params):
        peakStress, peakStrain, modulus, alphaScale = params
        return CDPCostAlphaScale(peakStress, peakStrain, modulus, alphaScale, expStrain, expStress)
    peakIndex = np.argmax(expStress)
    stressScale = expStress[peakIndex]
    strainScale = expStrain[peakIndex]
    stressList = expStress/stressScale
    strainList = expStrain/strainScale
    if(initial_params and len(initial_params)==4):
        peakStressInitial, peakStrainInitial, modulusInitial, alphaInitial = initial_params 
    else:
        peakStressInitial = 1.0
        peakStrainInitial = 1.0
        modulusInitial = 1.5
        alphaInitial = 1.0
    cons = ({'type': 'ineq', 'fun': lambda x:  0.5-abs(x[0]-1) },
            {'type': 'ineq', 'fun': lambda x:  0.5-abs(x[1]-1) },
            {'type': 'ineq', 'fun': lambda x:  x[2] },
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[0]/x[1] },
            {'type': 'ineq', 'fun': lambda x:  x[3]+100},
            {'type': 'ineq', 'fun': lambda x:  100-x[3]})
    res = minimize(objective, x0=[peakStressInitial, peakStrainInitial, modulusInitial, alphaInitial], constraints=cons )
    return res, strainScale, stressScale

def fittingCDPonXLSX(fileName):
    #fileName = "0.5-1-400P TestNo=584.xlsx"
    pdData = pd.read_excel(fileName)
    strainList = pdData.values[:,-2]*0.1
    stressList = pdData.values[:,-1]
    peakIndex = np.argmax(stressList)
    stressScale = stressList[peakIndex]
    strainScale = strainList[peakIndex]
    stressList = stressList/stressScale
    strainList = strainList/strainScale
    #fig, ax = plt.subplots()
    #ax.plot(strainList, stressList)
    
    fun = lambda x: CDPCostAlphaScale(x[0], x[1], x[2], x[3], strainList, stressList)
    peakStressInitial = 1.0
    peakStrainInitial = 1.0
    modulusInitial = 1.5
    alphaInitial = 1.0
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] },
            {'type': 'ineq', 'fun': lambda x:  x[1] },
            {'type': 'ineq', 'fun': lambda x:  x[2] },
            #{'type': 'ineq', 'fun': lambda x:  x[2] - x[0]/x[1] },
            {'type': 'ineq', 'fun': lambda x:  x[3]+10},
            {'type': 'ineq', 'fun': lambda x:  10-x[3]})
    res = minimize(fun, x0=[peakStressInitial, peakStrainInitial, modulusInitial, alphaInitial], constraints=cons )
    #print(res.x)
    #strainMini, stressMini = CDPStressAlphaScale(res.x[0], res.x[1], res.x[2], res.x[3], strainList)
    #ax.plot(strainMini, stressMini)
    #plt.show()
    return res, stressScale, strainScale

