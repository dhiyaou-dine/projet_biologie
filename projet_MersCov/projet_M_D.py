from __future__ import print_function
import openturns as ot
import math as m
import openturns.viewer as otv
from numpy import zeros,array
import SIR_enonce as SIR
import SIR_enonce2 as SIR2
import math
## modele DIR
# Set the ODE solver parameters
ot.ResourceMap.SetAsScalar("Fehlberg-InitialStep", 1.0e-1)
ot.ResourceMap.SetAsScalar("Fehlberg-LocalPrecision", 1.0e-6)
ot.ResourceMap.SetAsUnsignedInteger("Fehlberg-DefaultOrder", 4)

def buildFunction(beta, nu, N, t0):
    '''
       Build the transition function of the SIR model given the values of
       beta and nu
    '''
    f = ot.SymbolicFunction(['t', 'S', 'I', 'R'], [str(-beta/N)  + '*S*I', str(beta/N) + 'S*I-' + str(nu) + '*I', str(nu) + '*I'])
    phi = ot.ParametricFunction(f, [0], [t0])
    return phi
    

def computeMax(X):
    '''
       Compute the maximum number of people in group I over [0,T]
       Note: the ODE is integrated using an adaptive method, and the
       measurement is done with a time step of one hour
    '''
    dt = 1.0 / 24.0
    N0 = X[0]
    I0 = X[1]
    betaIni = X[2]
    betaNew = X[3]
    tNew = X[4]
    nu = X[5]
    # First phase, no intervention
    phi = buildFunction(betaIni, nu, N0, 0.0)
    solver = ot.Fehlberg(phi)
    initialState = [N0-I0, I0, 0.0]
    nt = int(tNew / dt) + 1
    dt = tNew / (nt - 1)
    grid = ot.RegularGrid(0.0, dt, nt).getVertices().asPoint()
    result = solver.solve(initialState, grid)
    # Extract the maximum number of infected people in phase 0
    max0 = result.getMax()[1]
    # Second phase, after intervention
    phi = buildFunction(betaNew, nu, N0, tNew)
    solver = ot.Fehlberg(phi)
    initialState = result[-1]
    grid = ot.RegularGrid(tNew, dt, int(300 / dt + 1)).getVertices().asPoint()
    result = solver.solve(initialState, grid)
    # Extract the maximum number of infected people in phase 1
    max1 = result.getMax()[1]
    return [max(max0, max1)]

## fin modele SIR
##modele SIR2
# Set the ODE solver parameters
ot.ResourceMap.SetAsScalar("Fehlberg-InitialStep", 1.0e-1)
ot.ResourceMap.SetAsScalar("Fehlberg-LocalPrecision", 1.0e-6)
ot.ResourceMap.SetAsUnsignedInteger("Fehlberg-DefaultOrder", 4)

def buildFunction(beta, nu, N, t0, r):
    '''
       Build the transition function of the SIR model given the values of
       beta and nu
    '''
    f = ot.SymbolicFunction(['t', 'S', 'I', 'R'], [str(-beta/N)  + '*S*I' + '*r', str(beta/N) + 'S*I-' + str(nu) + '*I' + '*r', str(nu) + '*I' + '*r'])
    phi = ot.ParametricFunction(f, [0], [t0])
    return phi
    

def computeMaxFuzzy(X):
    '''
       Compute the maximum number of people in group I over [0,T]
       Note: the ODE is integrated using an adaptive method, and the
       measurement is done with a time step of one hour
    '''
    dt = 1.0 / 24.0
    N0 = X[0]
    I0 = X[1]
    betaIni = X[2]
    betaNew = X[3]
    tNew = X[4]
    nu = X[5]
    r  = X[6]
    # First phase, no intervention
    phi = buildFunction(betaIni, nu, N0, 0.0, r)
    solver = ot.Fehlberg(phi)
    initialState = [N0-I0, I0, 0.0]
    nt = int(tNew / dt) + 1
    dt = tNew / (nt - 1)
    grid = ot.RegularGrid(0.0, dt, nt).getVertices().asPoint()
    result = solver.solve(initialState, grid)
    # Extract the maximum number of infected people in phase 0
    max0 = result.getMax()[1]
    # Second phase, after intervention
    phi = buildFunction(betaNew, nu, N0, tNew, r)
    solver = ot.Fehlberg(phi)
    initialState = result[-1]
    grid = ot.RegularGrid(tNew, dt, int(300 / dt + 1)).getVertices().asPoint()
    result = solver.solve(initialState, grid)
    # Extract the maximum number of infected people in phase 1
    max1 = result.getMax()[1]
    return [max(max0, max1)]

##fin modele SIR2

# la population du Corée du sud suit la loi U(50 950 000, 51 050 000)
    
N0 =  ot.Uniform(50950000, 51050000)

graph1 = N0.drawPDF()
graph1.setLegends("N0")
graph1.setTitle("loi uniforme de N0")
#ot.Show(graph1)

#la population infectious I(0) distribué selon une loi Beta de paramètres B(2, 5, 1, 3)

I0 = ot.Beta(2, 5, 1, 3)

# source d'incertitudes (le vecteur aléatoire (Beta,nu)
# βini suit une loi Beta B(5.5, 5.5, 0.215, 0.248)

betaIni = ot.Beta(5.5, 5.5, 0.215, 0.248)
    
# ν suit une loi Gamma de paramètres G(7.3, 975, 0.021)

nu = ot.Gamma(7.3, 975.0, 0.021)
graph2 = nu.drawPDF( )
graph2.setLegends("nu")
graph2.setTitle("loi gamma de nu")
#ot.Show(graph2)

# le taux de reproduction de base R0

h = ot.SymbolicFunction(["x1", "x2"], [" x1/x2"])

#le vecteur aléa d'entrée

dist1 = ot.ComposedDistribution([betaIni, nu])
graph3 = dist1.drawPDF()
graph3.setLegends("R0")
graph3.setTitle("densité de R0")
#ot.Show(graph3)

X1 = ot.RandomVector(dist1)

#le vecteur de sortie

R0 = ot.CompositeRandomVector(h, X1)

#Calculde probabilité R0<1

#Evénement
s = 1.0
test = ot.Less()
E = ot.ThresholdEvent(R0, test, s)
# Define a solver
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(1000)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
# Run FORM
algo = ot.FORM(optimAlgo, E, dist1.getMean())
algo.run()
result = algo.getResult()

# Probability
pIni = result.getEventProbability()
print("la probabilité est de pIni =  ",pIni)
# prise en compte des mesures de controle d l'epidemie

#loi uniforme du nouveau temps U(24; 29)
Tnew =  ot.Uniform(24, 29)

#loi gamma du nouveau beta G(2:8; 2870; 0:0183)
betaNew = ot.Gamma(2.8, 2870.0, 0.0183)

#le vecteur aléa d'entrée

dist2 = ot.ComposedDistribution([betaNew, nu])
graph4 = dist1.drawPDF()
graph4.setLegends("R0")
graph4.setTitle("densité de R0")
ot.Show(graph4)

X2 = ot.RandomVector(dist2)

#le vecteur de sortie

R0New = ot.CompositeRandomVector(h, X2)

#Calculde probabilité R0<1

#Evénement
s = 1.0
test = ot.Less()
ENew = ot.ThresholdEvent(R0New, test, s)
# Run FORM
algoNew = ot.FORM(optimAlgo, ENew, dist2.getMean())
algoNew.run()
resultNew = algoNew.getResult()

# Probability
pNew = resultNew.getEventProbability()
print("la probabilité est de pNew =  ",pNew)
# modele aléatoire
#Créer le vecteur X = (N(0), I(0), βini, βnew, tnew, ν) des sources d’incertitudes
marginales = [N0, I0, betaIni, betaNew, Tnew, nu]
dist3 = ot.ComposedDistribution(marginales)
X = ot.RandomVector(dist3)
#creation de la variable interet
f = ot.PythonFunction(6, 1, SIR.computeMax, n_cpus=-1)
Y = ot.CompositeRandomVector(f, X)
#sample de n points : echantillon
n = 10000
sample = Y.getSample(n) 
#estimation de la moyenne et de l'écart type 

kernel = ot.KernelSmoothing()
estimated = kernel.build(sample)
print("Mean = ", sample.computeMean())

print("Ecart-type = ",sample.computeVariance()  )

sample.exportToCSVFile('sample.csv', '; ')
otv.View(estimated.drawCDF()).save ("estimate.png")
#calcul de probabilité d'evenement rare
#Calcul de probabilité d’événements rares
s2 = 1000.0
test2 = ot.Greater() 
G = ot.ThresholdEvent(Y, test2, s2)
experiment = ot.MonteCarloExperiment()
myAlgo = ot.ProbabilitySimulationAlgorithm(G, experiment)
myAlgo.setMaximumOuterSampling(100)
myAlgo.setBlockSize(4) # nombre d'estimations reparties entre 100 et 4 pour 400
#myAlgo.setMaximumCoefficientOfVariation(0.1)
# Perform the simulation
myAlgo.run()
print("variation = ",myAlgo.getMaximumCoefficientOfVariation())
print('Probability estimateMontCarlo=%.6f' % myAlgo.getResult().getProbabilityEstimate())
#methode FORM pour approximer la probabilité de l'evenement E
#Evénement
    #defini lors du calcu du proba avec montecarlo 
# Run FORM
algoFORM = ot.FORM(optimAlgo, G, dist3.getMean())
algoFORM.run()
resultFORM = algoFORM.getResult()
# Probability
pFORM = resultFORM.getEventProbability()
print("la probabilité avec forme est pFORME = ", pFORM) 


#le modéle SIR modifié
#loi beta de r B(2; 4; 0:9; 1:2)



r = ot.Beta(2.0, 4.0, 0.9, 1.2)
r.setDescription(["r"])
marginales_r = list(marginales)
marginales_r.append(r)
dist_X_r = ot.ComposedDistribution(marginales_r)#xTild
XTild = ot.RandomVector(dist_X_r)
#modéle augmenté fTild =r*f
fTild = ot.PythonFunction(7, 1, SIR2.computeMaxFuzzy)
YTild = ot.CompositeRandomVector(fTild, XTild)

#evenement Etild situation redoutée
GTild = ot.ThresholdEvent(YTild, test2, s2)

# Run FORM
algoFORMTild = ot.FORM(optimAlgo, GTild, dist_X_r.getMean())
algoFORMTild.run()
resultFORMTild = algoFORMTild.getResult()
# Probability
pFORMTild = resultFORMTild.getEventProbability()
print("la probabilité avec forme est pFORMTild = ", pFORMTild) 

#analyse de sensibilité
#calcul des indices de sobol

input_dimension = dist_X_r.getDimension()
size = 100
degree = 3
basisSize = ot.LinearEnumerateFunction(input_dimension).getStrataCumulatedCardinal(degree)
projection = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
inputSample = dist_X_r.getSample(size)
outputSample = fTild(inputSample)
basis = ot.OrthogonalProductPolynomialFactory([ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(distM)) for distM in marginales_r])
algo_meta = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_X_r, ot.FixedStrategy(basis, basisSize), projection)
algo_meta.run()
result_meta = algo_meta.getResult()
meta_modele = result_meta.getMetaModel()
#print(meta_modele)
inputValidation = dist_X_r.getSample(size)
outputValidation = fTild(inputValidation)
validation = ot.MetaModelValidation(inputValidation, outputValidation, meta_modele)
graph = validation.drawValidation()
ot.Show(graph)
#monte carlo sur meta_model
CoV = 0.01
Y_meta = ot.CompositeRandomVector(meta_modele, XTild)
event_meta = ot.ThresholdEvent(Y_meta, ot.Less(), 0.0)
algo = ot.ProbabilitySimulationAlgorithm(event_meta, ot.MonteCarloExperiment())
algo.setMaximumCoefficientOfVariation(CoV)
algo.setMaximumOuterSampling(1000)
algo.setBlockSize(1000)
algo.run()
result = algo.getResult()
print("p=%.2e", result.getProbabilityEstimate())
print("CoV=%.2g",  result.getCoefficientOfVariation())
print("N=", result.getOuterSampling())
#indice de sobol   
post_processing = ot.FunctionalChaosSobolIndices(result_meta)
sobol = ot.Sample(1, input_dimension)
sobol.setDescription(dist_X_r.getDescription())
for j in range(input_dimension):
    sobol[0,j]=100.0 * post_processing.getSobolIndex(j)
print("Sobol indices ",sobol)
#indice de sobol totaux
sobolT = ot.Sample(1, input_dimension)
sobolT.setDescription(dist_X_r.getDescription())
for j in range(input_dimension):
    sobolT[0,j]=100.0 * post_processing.getSobolTotalIndex(j)
    #print("Sobol total indices", sobolT)
#Estimation esperance de XTild/ETild
sample_X = XTild.getSample(size)
sample_Y = meta_modele(sample_X)
conditional_sample = ot.Sample(1, sample_X.getDimension())
for j in range(size):
    if sample_Y[j, 0] < 0.0:
        conditional_sample.add(sample_X[j])
#print("size=", conditional_sample.getSize())
# Esperance conditionnelle
esperance_conditionnelle = conditional_sample.computeMean()
print("E[X|f(X)\\in D_f]= ", esperance_conditionnelle)
#Facteur d'inportance
# Facteurs d'importance dans l'espace standard\n",
u = dist_X_r.getIsoProbabilisticTransformation()(conditional_sample).computeMean()
IF = ot.Sample(1, input_dimension)
IF.setDescription(dist_X_r.getDescription())
for j in range(input_dimension):
    IF[0, j] = (100.0 * u[j]**2 / u.normSquare())
print("Importance factors", IF)
    







