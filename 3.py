import json
import numpy as np


def wolfe_inexact_line_search(function, gradient, xk, pk):
    """
        Fun fact: as I understood this is already implemented in scipy
        Calling sp.optimize.line_search(function, gradient, xk, pk)[0] returns the same final answer (but different number of iterations)
    """
    delta1, delta3 = 0.4, 0.6
    theta1, theta2 = 1.5, 0.5
    tau_l, tau_r = 0, 0
    np.random.seed(1)
    tau = np.random.rand()

    fk = function(xk)
    dfk = gradient(xk)
    iter_count = 0
    while True:
        fk1 = function(xk + tau * pk)
        dfk1 = gradient(xk + tau * pk)
        condition1 = (fk1 <= fk + delta1 * np.dot(dfk, pk) * tau)
        condition2 = (np.dot(dfk1, pk) >= delta3 * np.dot(dfk, pk))
        if not condition1:
            tau_r = tau
            tau = (1 - theta2) * tau_l + theta2 * tau_r
        elif not condition2:
            tau_l = tau
            if tau_r == 0:
                tau *= theta1
            else:
                tau = (1 - theta2) * tau_l + theta2 * tau_r
        else:
            break
        iter_count += 1
    return tau


def bfgs(function, gradient, x0, epsilon):
    gfk = gradient(x0)
    I = np.eye(len(x0), dtype=int)
    Bk = I
    xk = x0
    k = 0
    while np.linalg.norm(gfk) > epsilon:
        pk = -np.dot(Bk, gfk)
        alpha_k = wolfe_inexact_line_search(function, gradient, xk, pk)

        sk = alpha_k * pk
        xk1 = xk + sk

        gfk1 = gradient(xk1)
        yk = gfk1 - gfk

        TMP1 = I - (sk[:, np.newaxis] * yk[np.newaxis, :]) / np.dot(yk, sk)
        TMP2 = I - (yk[:, np.newaxis] * sk[np.newaxis, :]) / np.dot(yk, sk)
        Bk = np.dot(TMP1, np.dot(Bk, TMP2)) + (sk[:, np.newaxis] * sk[np.newaxis, :]) / np.dot(yk, sk)

        xk, gfk = xk1, gfk1
        k += 1

    return {
        "Point": {f"x{i + 1}": xi for i, xi in enumerate(xk)},
        "Function value": function(xk),
        "Number of iterations": k
    }


def f(x):
    first_term = sum(map(lambda xi: (xi ** 2 - 2) ** 2, x[:-1]))
    second_term = (sum(map(lambda xi: xi ** 2, x)) - 0.5) ** 2
    return first_term + second_term


def grad(x):
    res = [0] * len(x)
    for i in range(len(x) - 1):
        res[i] += 2 * (x[i] ** 2 - 2) * 2 * x[i]

    second_term = sum(map(lambda xi: xi ** 2, x)) - 0.5
    for i in range(len(x)):
        res[i] += 2 * second_term * 2 * x[i]

    return np.array(res)


x0 = np.array([1] * 10)
results = bfgs(function=f, gradient=grad, x0=x0, epsilon=0.001)
print(json.dumps(results, indent=4))
