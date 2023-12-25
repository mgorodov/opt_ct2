from math import sin, cos
import json
import numpy as np


def goldstein_inexact_line_search(function, gradient, xk, pk, epsilon=0.000001):
    delta1, delta2 = 0.4, 0.6
    theta1, theta2 = 1.5, 0.5
    tau_l, tau_r = 0, 0
    np.random.seed(69)
    tau = np.random.rand()

    fk = function(xk)
    dfk = gradient(xk)
    iter_count = 0
    while True:
        fk1 = function(xk + tau * pk)
        condition1 = (fk1 <= fk + delta1 * np.dot(dfk, pk) * tau + epsilon)
        condition2 = (fk1 >= fk + delta2 * np.dot(dfk, pk) * tau - epsilon)
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


def fletcher_reeves(function, gradient, x0, epsilon):
    k = 0
    pk = -gradient(x0)
    xk = x0
    while np.linalg.norm(gradient(xk)) > epsilon:
        alpha = goldstein_inexact_line_search(function, gradient, xk, pk)
        xk1 = xk + alpha * pk

        beta = np.linalg.norm(gradient(xk1)) ** 2 / np.linalg.norm(gradient(xk)) ** 2
        pk = -gradient(xk1) + beta * pk

        xk = xk1
        k += 1
    return {
        "Point": {f"x{i + 1}": xi for i, xi in enumerate(xk)},
        "Function value": function(xk),
        "Number of iterations": k
    }


def f(x):
    res = sum(map(lambda xi: sin(x[0] + xi ** 2 - 1), x[:-1]))
    res += 0.5 * sin(x[-1] ** 2)
    return res


def grad(x):
    res = [0] * len(x)
    for i in range(len(x) - 1):
        res[0] += cos(x[0] + x[i] ** 2 - 1) * 1
        res[i] += cos(x[0] + x[i] ** 2 - 1) * 2 * x[i]
    res[-1] += 0.5 * cos(x[-1] ** 2) * 2 * x[-1]
    return np.array(res)


x0 = np.array([1] * 10)
results = fletcher_reeves(function=f, gradient=grad, x0=x0, epsilon=0.001)
print(json.dumps(results, indent=4))
