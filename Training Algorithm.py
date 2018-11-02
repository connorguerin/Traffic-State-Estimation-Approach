import numpy as np
from DataGenerator import DataGenerator
from sklearn.linear_model import LinearRegression

# tolerance for convergence of soln
error = 0.01
# specified mu for phi(x)
mu = 0.01
#observation and feature numbers
n = 10
p = 3


def main():
    #generate data points
    data = DataGenerator(n, p)
    X, y, b = data.generatePoints()

    # construct constraint matrix A
    A = constraintMatrix(n)

    # construct initial point
    #x0 = np.array(initial_point(X, y))
    x0 = np.ones([4*p, 1])
    #resize to account for standard form variables
    #x0.resize(4*p, 1)

    #calculate first search direction
    cfs = closedFormSolution(x0, A, mu, X, y, b)
    x1 = x0 + cfs

    counter = 0
    #loop until specified tolerance is met
    while (abs((Fx(x1, X, y, b) - Fx(x0, X, y, b))) > error):
        print(counter)
        x0 = x1
        x1 = x1 + closedFormSolution(x1, A, mu, X, y, b)




#function for calculating dt
def closedFormSolution (x, A, mu, observations, responses, true):
    x = np.array(x)
    X = np.diagflat(x)
    A = np.array(A)
    e = np.ones(x.size)
    b = beta(mu)
    print(X)
    gradientPhi = gradientFx(x, observations, responses, true) - mu * np.matmul(np.linalg.inv(X), e)
    pterm1 = np.linalg.inv(np.matmul(np.matmul(A,np.matmul(X,X)),A.T))
    pterm2 = np.matmul(np.matmul(A,np.matmul(X,X)), gradientPhi)
    p = np.matmul(pterm1, pterm2)
    denom = np.linalg.norm(np.matmul(X, (gradientPhi - np.matmul(A.T, p))))
    if (denom != 0):
        num = np.matmul(np.matmul(X,X), (gradientPhi - np.matmul(A.T, p)))
        return -b*(num/denom)
    else:
        return 0


#method for defining function f(x)
def Fx(xt, observations, responses, b):
    obs = observations
    resp = responses
    xtrue = b
    delta = 0.01
    gamma = 0.2
    lam = 0.5
    f = 1/(2*n)*np.linalg.norm((np.matmul(obs, xt[:p])-resp))**2
    f += delta * (np.linalg.norm((xt[:p] - xtrue))**2)
    mcp = 0
    for xi in range(p*3, len(xt)):
        if (xi <= gamma*lam):
            mcp += lam*xi-(xi**2)/(2*gamma)
        else:
            mcp += (1/2)*gamma*(lam**2)
    f += mcp
    return f

#method for defining the gradient of the objective function
def gradientFx (xt, observations, responses, b):
    obs = observations
    resp = responses
    xtrue = b
    delta = 0.01
    gamma = 0.2
    lam = 0.5

    fprime = (1/n) * np.matmul(obs.T, (np.matmul(obs, xt[:p]) - resp))
    fprime += 2* delta * (xt[:p] - xtrue)
    mcp = 0
    for xi in range(p*3, len(xt)):
        if (xi <= gamma*lam):
            mcp += (lam - xi/gamma)*np.sign(xi)

    fprime += mcp
    return fprime

#method calculates beta
def beta (epsilon):
    #TODO: determine Lipschitz constant based on gradient of function
    L = 50
    return epsilon/(L + 2 * epsilon)

def constraintMatrix (n):
    constraint_one = np.zeros([n * 4])
    constraint_one[0] = 1
    constraint_one[n] = 1
    constraint_one[n * 3] = -1

    constraint_two = np.zeros([n * 4])
    constraint_two[0] = -1
    constraint_two[n * 2] = 1
    constraint_two[n * 3] = -1

    A = np.array(constraint_one)
    B = np.array(constraint_one)
    C = np.array(constraint_two)

    for i in range(0, n - 1):
        for j in range(n * 4 - 1, -1, -1):
            if B[j] <> 0:
                B[j + 1] = B[j]
                B[j] = 0
        A = np.vstack((A, B))

    A = np.vstack((A, C))

    for i in range(0, n - 1):
        for j in range(n * 4 - 1, -1, -1):
            if C[j] <> 0:
                C[j + 1] = C[j]
                C[j] = 0
        A = np.vstack((A, C))

    return A

def initial_point(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg.coef_

if __name__ == '__main__':
    main()
