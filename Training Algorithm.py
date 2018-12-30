import numpy as np
from DataGenerator import DataGenerator

# tolerance for convergence of soln
error = 0.1
# specified mu for phi(x)
mu = 0.1

L = 5
b = mu / (L + 2 * mu)
#observation and feature numbers
n = 500
p = 1000
#generate data points
data = DataGenerator(n, p)
X, y, betaTrue = data.generatePoints()

def main():
    global b
    # construct constraint matrix A
    A = constraintMatrix()
    # construct initial point
    x0 = np.array(initial_point())
    #resize to account for slack variables
    #calculate first search direction
    cfs = np.array(closedFormSolution(x0, A))
    x1 = np.add(x0,np.matmul(np.diagflat(x0),cfs))
    counter = 0
    #loop until specified tolerance is met
    while (abs(Fx(x1)-Fx(x0)) > error):
            if (Fx(x1) > Fx(x0)):
                b /= 2
            x0 = x1
            d = closedFormSolution(x0, A)
            x1 = np.add(x1, np.matmul(np.diagflat(x1), d))
            # if (np.linalg.norm(d) <= b ):
            #     b /= 2
            counter+=1
            #print(abs(Fx(x1)-Fx(x0)))
            print(str(abs(Fx(x1) - Fx(x0))) + " " + str(counter))
            if counter >= 1000:
                break
    print("MSE of true term and OLS solution")
    print(round(np.square(np.subtract(initial_point()[:p], betaTrue)).mean(), 6))
    print("MSE of true term and algorithm solution")
    print(round(np.square(np.subtract(x1[:p], betaTrue)).mean(), 6))
    print(x1[:p])
    print(betaTrue)
    # print("iterations:" + str(counter))
    #print(Fx(x0))
    #print(Fx(x1))


def closedFormSolution (x, A):
    A=np.array(A)
    x = np.array(x)
    diagX = np.diagflat(x)
    e = np.int8(np.ones(x.size))
    barrier = np.fromiter((term*mu for term in np.matmul(np.linalg.inv(diagX), e)), dtype=float)
    gradientPhi = np.subtract(gradientFx(x).T, barrier)
    for i in range(0, x.size):
        diagX[i][i] = diagX[i][i] ** 2
    term = np.matmul(A, diagX)
    p = np.linalg.inv(np.matmul(term, A.T))
    l = np.matmul(term, gradientPhi.T)
    p = np.matmul(p, l)
    denom = np.matmul(np.diagflat(x), np.subtract(gradientPhi.T, np.matmul(A.T, p)))
    if (np.linalg.norm(denom) != 0):
        num = np.matmul(diagX, denom)
        div = np.linalg.norm(denom)
        num = [(aTerm * b * (-1)) / div for aTerm in num]
        return num
    else:
        return 0


#method for defining function f(x)
def Fx(xt):
    lam = 500
    f = float(1)/(2*n)*(np.linalg.norm(np.subtract(np.matmul(X, xt[:p]),y))**2) + np.linalg.norm(xt[:p], ord=1)*lam
    return f

#method for defining the gradient of the objective function
def gradientFx (xt):
    lam = 500

    fPrime = np.subtract(np.matmul(np.matmul(X.T,X), xt[:p]), np.matmul(X.T,y))
    fPrime = np.array([x * 2  for x in fPrime])
    norm = np.array(xt[:p])
    for i in range(0, norm.size):
        if abs(xt[i]) > 0:
            norm[i] = np.sign(xt[i])*lam

    fPrime = np.add(fPrime, norm)
    fPrime.resize(p*2, 1, refcheck=False)

    return fPrime

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
    point =  np.full([p,1], 5)
    point.resize(2 * p, 1, refcheck=False)
    for i in range(0, p):
        # x0[i] = abs(x0[i])
        point[i + p] = point[i] * -1
    return point

if __name__ == '__main__':
    main()
