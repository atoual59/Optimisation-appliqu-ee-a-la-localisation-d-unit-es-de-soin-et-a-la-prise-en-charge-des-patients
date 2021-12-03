
#!/usr/bin/python

# Copyright 2013, Gurobi Optimization, Inc.


from gurobipy import *

import numpy as np

populations=np.array([479553,340017,309346,285121,280966,254436,232787,216815,182460,172565,171953,170147,158454,156920,152960])

distance=np.array([[0,562,585,242,949,245,894,697,812,536,469,848,532,644,635],
                   [562,0,1143,326,790,803,1157,1186,955,490,149,1119,465,660,1062],
                   [585,1143,0,824,860,347,597,107,515,662,1051,384,786,638,88],
                   [242,326,824,0,791,482,963,894,787,322,233,917,297,492,771],
                   [949,790,860,791,0,942,549,827,347,550,866,695,533,332,773],
                   [245,803,347,482,942,0,799,458,717,530,711,651,654,640,396],
                   [894,1157,597,963,549,799,0,572,199,749,1065,316,803,501,511],
                   [697,1186,107,894,827,458,572,0,483,699,1092,280,823,620,126],
                   [812,955,515,787,347,717,199,483,0,546,862,344,600,298,429],
                   [536,490,662,322,550,530,749,699,546,0,398,720,155,251,574],
                   [469,149,1051,233,866,711,1065,1092,862,398,0,1026,372,567,969],
                   [848,1119,384,917,695,651,316,280,344,720,1026,0,767,506,295],
                   [532,465,786,297,533,654,803,823,600,155,372,767,0,304,699],
                   [644,660,638,492,332,640,501,620,298,251,567,506,304,0,551],
                   [635,1062,88,771,773,396,511,126,429,574,969,295,699,551,0]])
regions= np.array(['toulouse','nice','nantes','montpelier','strasbourg','bordeaux','lille','rennes','reims','saint-etiennes','toulon','le havre','grenoble','dijon','angers'])



k=5
alpha=0.2
coeff=((1+alpha)/k)*np.sum(populations)

nbreg=regions.shape[0]

nbcont=nbreg+2
nbvar=(nbreg+1)*nbreg

# Range of plants and warehouses
lignes = range(nbcont)
colonnes = range(nbreg)

# Matrice des contraintes
a = np.zeros(((nbcont),nbreg))
a=a.astype(int)
for i in range (a.shape[0]):
    for j in range (a.shape[1]):
        if (i<((a.shape[0])-2)):
            a[i][j]+=1
        if (i==((a.shape[0])-2)):
            a[i][j]= populations[j]
        if (i==((a.shape[0])-1)):
            a[i][j]= 1
        

# Second membre
b = np.zeros(nbcont) +1
b=b.astype(int)
b[nbcont-2]=coeff
b[nbcont-1]=k

# Coefficients de la fonction objectif
c = np.zeros((nbreg,nbreg))+1
for i in range (c.shape[0]):
    for j in range (c.shape[1]):
        c[i][j]=c[i][j]*distance[i][j]*populations[i]


c=np.reshape(c,c.size)

m = Model("mogplex")     
        
# declaration variables de decision
x = []
for i in range (nbvar):
    x.append(m.addVar(vtype=GRB.BINARY, lb=0, name="x%d" % (i+1) ))


# maj du modele pour integrer les nouvelles variables
m.update()

obj = LinExpr();
obj =0
for j in range(nbvar-nbreg):
    l=int(j%nbreg) +1
    obj += c[j] * x[j] * x[(nbvar-nbreg-1+l)]
# definition de l'objectif
m.setObjective(obj,GRB.MINIMIZE)

# Definition des contraintes
for i in lignes:
    if (i<nbreg):  
        m.addConstr(quicksum(x[j+(i*nbreg)]*x[nbvar-nbreg+j] for j in colonnes) == b[i], "Contrainte%d" % i)
    if (i==nbreg):
        m.addConstr(quicksum(a[i][j]*x[nbvar-nbreg+j] for j in colonnes) <= b[i], "Contrainte%d" % i)
    if (i>nbreg):
        m.addConstr(quicksum(x[nbvar-nbreg+j] for j in colonnes) == b[i], "Contrainte%d" % i)
   

# Resolution
m.optimize()


print('lensemble des centre est : ')
for j in range(15):
    ctr=(nbreg*nbreg)+j
    centre=x[ctr].x
    if (centre):
        print(regions[j])
print('')
for j in range(nbvar-nbreg):
    reg=int(j/nbreg)
    ct=j%nbreg
    if (x[j].x):
        print(regions[reg],'est affectée au centre de',regions[ct])
print("")
print('la somme de la distance totale est:', m.objVal)