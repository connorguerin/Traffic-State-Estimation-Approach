import numpy as np
from DataGenerator import DataGenerator

# tolerance for convergence of soln
error = 0.000001
# specified mu for phi(x)
mu = 0.01
#observation and feature numbers
n = 5
p = 10
#generate data points
data = DataGenerator(n, p)
X, y, b = data.generatePoints()

def main():
    global mu
    # construct constraint matrix A
    A = constraintMatrix()
    # construct initial point
    x0 = initial_point()
    print(np.diagflat(x0))
    #resize to account for slack variables
    x0.resize(2*p, 1, refcheck=False)
    for i in range(0, p):
        x0[i + p] = x0[i] * -1
    #calculate first search direction
    cfs = np.array(closedFormSolution(x0, A))
    x1 = x0 + np.matmul(np.diagflat(x0),cfs)
    counter = 0
    #loop until specified tolerance is met
    while (abs((Fx(x1) - Fx(x0))) > error):
            if (Fx(x1) > Fx(x0)):
                mu /= 2
            x0 = x1
            x1 = x1 + np.matmul(np.diagflat(x1),closedFormSolution(x1, A))
            counter += 1

    print("MSE of true term and OLS solution")
    print(round(np.square(np.subtract(initial_point(), b)).mean(), 6))
    print("MSE of true term and algorithm solution")
    print(round(np.square(np.subtract(x1[:p], b)).mean(), 6))
    print(x1[:p])
    print(b)
    print(initial_point())
    print("iterations:" + str(counter))

def closedFormSolution (x, A):
    A=np.round(np.array(A), 3)
    x = np.array(x)
    diagX = np.diagflat(x)
    e = np.int8(np.ones(x.size))
    b = beta(mu)
    barrier = np.fromiter((term*mu for term in np.matmul(np.linalg.inv(diagX), e)), dtype=float)
    gradientPhi = np.subtract(gradientFx(x).T, barrier)
    p = np.matmul(np.linalg.inv(np.matmul(np.matmul(A,np.matmul(diagX,diagX)),A.T)), np.matmul(np.matmul(A,np.matmul(diagX,diagX)), gradientPhi.T))
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

    fPrime = np.subtract(np.matmul(np.matmul(X.T,X), xt[:p]), np.matmul(X.T,y))
    fPrime = np.array([x * 2 for x in fPrime])
    fPrime = np.add(fPrime, 2* delta * (xt[:p] - b))
    mcp = np.zeros(p)
    for xi in range(0, p):
        if (abs(xt[xi]) <= gamma*lam):
            mcp[xi] = (lam - abs(xt[xi])/gamma)*np.sign(xt[xi])

    fPrime = np.add(fPrime.T, mcp)
    #resize for slack variables
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
    A = np.empty((p, 2 * p))
    A[0] = np.array(constraint_one)
    B = np.array(constraint_one)
    for i in range(0, p - 1):
        for j in range(p * 2 - 1, -1, -1):
            if B[j] <> 0:
                B[j + 1] = B[j]
                B[j] = 0
        A[i + 1] = B
    return np.int8(A)

def initial_point():
    s, residuals, rank, sing = np.linalg.lstsq(X,y, rcond=None)
    return np.array(s)

if __name__ == '__main__':
    main()
