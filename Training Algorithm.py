import numpy as np
from DataGenerator import DataGenerator
from sklearn.linear_model import LinearRegression

# tolerance for convergence of soln
error = 0.00000001
# specified mu for phi(x)
mu = 0.01
#observation and feature numbers
n = 2
p = 5
#generate data points
data = DataGenerator(n, p)
X, y, b = data.generatePoints()

def main():
    # construct constraint matrix A
    A = constraintMatrix()
    # construct initial point
    x0 = np.absolute(initial_point())
    #resize to account for standard form variables
    x0.resize(2*p, 1, refcheck=False)
    for i in range(0, p):
        x0[i + p] = x0[i] * -1
    #calculate first search direction
    cfs = np.array(closedFormSolution(x0, A))
    x1 = x0 + cfs

    counter = 0
    #loop until specified tolerance is met
    while (abs((Fx(x1) - Fx(x0))) > error):
        if counter > 100000:
            print(counter)
            print(Fx(x1))
            break
        x0 = x1
        x1 = x1 + closedFormSolution(x1, A)
        counter += 1

    print(x1)
    print("\n")
    print(b)


#function for calculating dt
def closedFormSolution (x, A):
    x = np.array(x)
    diagX = np.diagflat(x)
    A = np.array(A)
    e = np.ones(x.size)
    b = beta(mu)

    penaltyTerm = np.array([term*mu for term in np.matmul(np.linalg.inv(diagX), e)])
    gradientPhi = np.subtract(gradientFx(x).T, penaltyTerm)
    pterm1 = np.linalg.inv(np.matmul(np.matmul(A,np.matmul(diagX,diagX)),A.T))
    pterm2 = np.matmul(np.matmul(A,np.matmul(diagX,diagX)), gradientPhi.T)
    p = np.matmul(pterm1, pterm2)
    denom = np.linalg.norm(np.matmul(diagX, (gradientPhi.T - np.matmul(A.T, p))))
    if (denom != 0):
        num = np.matmul(np.matmul(diagX,diagX), (gradientPhi.T - np.matmul(A.T, p)))
        num = [(aTerm * b * (-1))/denom for aTerm in num]
        return num

    else:
        return 0


#method for defining function f(x)
def Fx(xt):
    delta = 0.01
    gamma = 0.2
    lam = 0.5
    f = 1/(2*n)*np.linalg.norm((np.matmul(X, xt[:p])-y))**2
    f += delta * (np.linalg.norm((xt[:p] - b))**2)
    mcp = 0
    for xi in range(0, p):
        if (abs(xt[xi]) <= gamma*lam):
            mcp += lam*xt[xi]-(xt[xi]**2)/(2*gamma)
        else:
            mcp += (1/2)*gamma*(lam**2)
        f += mcp
    return f

#method for defining the gradient of the objective function
def gradientFx (xt):
    delta = 0.01
    gamma = 0.2
    lam = 0.5

    fPrime = np.matmul(X.T, (np.matmul(X, xt[:p]) - y))
    fPrime = np.array([x / n for x in fPrime])
    fPrime = np.add(fPrime, 2* delta * (xt[:p] - b))
    mcp = np.zeros(p)
    for xi in range(0, p):
        if (abs(xt[xi]) <= gamma*lam):
            mcp[xi] = (lam - abs(xt[xi])/gamma)*np.sign(xt[xi])

    fPrime = np.add(fPrime.T, mcp)
    fPrime.resize(p*2, 1, refcheck=False)

    return fPrime

#method calculates beta
def beta (epsilon):
    L = 50
    return epsilon/(L + 2 * epsilon)

def constraintMatrix ():
    constraint_one = np.zeros([p * 2])
    constraint_one[0] = 1
    constraint_one[p] = 1

    A = np.array(constraint_one)
    B = np.array(constraint_one)
    for i in range(0,p - 1):
        for j in range(p*2 - 1, -1, -1):
            if B[j] <> 0:
                B[j + 1] = B[j]
                B[j] = 0
        A = np.vstack((A, B))
    return A

def initial_point():
    reg = LinearRegression()
    reg.fit(X, y)

    return reg.coef_

if __name__ == '__main__':
    main()
