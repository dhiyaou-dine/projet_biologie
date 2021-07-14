import openturns as ot

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
