import concurrent
import concurrent.futures
from gurobipy import GRB, Model
from numpy import float128
import numpy as np
import matplotlib.pyplot as plt

NMAX = 7
mstar = 3
dc = mstar - 1
d = 5e2
lamb = 1.8319311506873706
n = d * 12
k = lamb * d
p2 = 0.31032505675007377
p3 = 0.16596557534368533
n1 = n
# n1 = 1e10

k = float128(k)
d = float128(d)
n = float128(n)
p2 = float128(p2)
p3 = float128(p3)
dc = float128(dc)

states = ["GOOD", "BAD", "OTHER", "BADEND", "OTHEREND", "GOODEND"]


def compute_uppers():
    uppers = {
        "BAD": {
            "BAD": 1 - 2 * ((k - d - 2) / (n * k)),
            "OTHER": (((d - 2) * (1 + p2)) + p2) / ((n * k)),
            "GOOD": (k + (d * (p2 - 1)) - (p2 + 2)) / (n * k),
            "BADEND": (k + 2 * p2 * d) / (n * k),
            "OTHEREND": 0,
            "GOODEND": 0,
        },
        "OTHER": {
            "BAD": 2 * dc * (k - 3) / (n * k),
            "OTHER": 1 - (k - d - 2) / (n * k),
            "GOOD": (3 * dc * (d - 1) * p3) / (n * k),
            "BADEND": 0,
            "OTHEREND": (k + 2 * p2 * d) / (n * k),
            "GOODEND": 0,
        },
        "GOOD": {
            "BAD": (2 * d * (1 + p2) - p2) / (n * k),
            "OTHER": ((d - 2) * (6 * p3 + 2 * p2)) / (n * k),
            "GOOD": 1 - (k - d - 2) / (n * k),
            "BADEND": 0,
            "OTHEREND": 0,
            "GOODEND": (k + 2 * p2 * d) / (n * k),
        },
        "BADEND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 1,
            "OTHEREND": 0,
            "GOODEND": 0,
        },
        "OTHEREND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 0,
            "OTHEREND": 1,
            "GOODEND": 0,
        },
        "GOODEND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 0,
            "OTHEREND": 0,
            "GOODEND": 1,
        },
    }
    return uppers


def compute_lowers():
    lowers = {
        "BAD": {
            "BAD": 1 - 2 * ((2 * (d * p2 - 1) + k - p2) / (n * k)),
            "OTHER": 0,
            "GOOD": (k - d - 2) / (n * k),
            "BADEND": (k - d - 2) / (n * k),
            "OTHEREND": 0,
            "GOODEND": 0,
        },
        "OTHER": {
            "BAD": 0,
            "OTHER": 1
            - ((dc * (3 * p3 * (d - 1) + 2 * k - 6) + k + 2 * p2 * d) / (n * k)),
            "GOOD": 0,
            "BADEND": 0,
            "OTHEREND": (k - d - 2) / (n * k),
            "GOODEND": 0,
        },
        "GOOD": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 1 - (k + 2 * d + p2 * (5 * d - 3) + p3 * (3 * d - 6)) / (n * k),
            "BADEND": 0,
            "OTHEREND": 0,
            "GOODEND": (k - d - 2) / (n * k),
        },
        "BADEND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 1,
            "OTHEREND": 0,
            "GOODEND": 0,
        },
        "OTHEREND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 0,
            "OTHEREND": 1,
            "GOODEND": 0,
        },
        "GOODEND": {
            "BAD": 0,
            "OTHER": 0,
            "GOOD": 0,
            "BADEND": 0,
            "OTHEREND": 0,
            "GOODEND": 1,
        },
    }

    return lowers


terminate_lower = (k - d - 2) / (n * k)
terminate_upper = (k + (2 * p2 * d)) / (n * k)


def __name(layer, state):
    return f"{state}_{layer}"


def initial_constraints(model, prob_vars, states):
    model.addConstr(sum(prob_vars[__name(0, state)] for state in states) == 1)
    model.addConstr(prob_vars[__name(0, "BADEND")] == 0)
    model.addConstr(prob_vars[__name(0, "GOODEND")] == 0)
    model.addConstr(prob_vars[__name(0, "OTHEREND")] == 0)
    model.addConstr(prob_vars[__name(0, "OTHER")] == 0)


def add_layers(model, prob_vars, uppers, lowers, states, l):
    p_vars = {__name(l, s): model.addVar(lb=0, ub=1, name=__name(l, s)) for s in states}

    model.addConstr(sum(p_vars.values()) == 1)

    for state in states:
        upper_bound = sum(
            uppers[s][state] * prob_vars[__name(l - 1, s)] for s in states
        )
        lower_bound = sum(
            lowers[s][state] * prob_vars[__name(l - 1, s)] for s in states
        )

        model.addConstr(p_vars[__name(l, state)] <= upper_bound)
        model.addConstr(p_vars[__name(l, state)] >= lower_bound)

    prob_vars.update(p_vars)


def get_probs(model, prob_vars, layers, states, _print=False):
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Optimization was unsuccessful")
        return float("inf"), float("inf"), float("inf"), float("inf")

    if _print:
        print("Optimization (Gamma): ", model.getVarByName("z_final").x)
        print("Optimization (Alpha): ", model.getVarByName("alpha").x)
        print("Optimization (Beta): ", model.getVarByName("beta").x)
        print("Optimization (Upsilon): ", model.getVarByName("upsilon").x)
        for state in states:
            print(__name(0, state), prob_vars[__name(0, state)].x)
            print(__name(layers, state), prob_vars[__name(layers, state)].x)

    return (
        prob_vars[__name(layers, "BAD")].x,
        prob_vars[__name(layers, "BADEND")].x,
        prob_vars[__name(layers, "GOOD")].x,
        prob_vars[__name(layers, "GOODEND")].x,
    )


def solve_layers(layers, uppers, lowers, states, _print=False):
    model = Model("LayerByLayer")
    model.setParam("OutputFlag", 0)

    prob_vars = {
        __name(0, s): model.addVar(lb=0, ub=1, name=__name(0, s)) for s in states
    }

    initial_constraints(model, prob_vars, states)

    for l in range(1, layers + 1):
        add_layers(model, prob_vars, uppers, lowers, states, l)

    z_final = model.addVar(lb=0, name="z_final")
    model.setObjective(z_final, GRB.MAXIMIZE)

    alpha = model.addVar(lb=0, name="alpha")  # alpha * p(goodend) - p(badend) = 0
    beta = model.addVar(lb=0, name="beta")
    upsilon = model.addVar(lb=0, name="upsilon")

    model.addConstr(
        beta
        == prob_vars[__name(layers, "BAD")]
        * (k - d - 2)
        * (k - d - 2)
        / (n1 * k * (k + d * (1 + 2 * p2) - 2))  # bad
        + (
            prob_vars[__name(layers, "GOOD")] * (k - d - 2) / (k + d * (1 + 2 * p2) - 2)
        )  # good
    )
    model.addConstr(upsilon * beta == 1)
    model.addConstr(
        alpha * prob_vars[__name(layers, "GOODEND")]
        - prob_vars[__name(layers, "BADEND")]
        == 0
    )  # alpha = badend / goodend

    prob_terminate = np.power(1 - terminate_lower, layers)
    model.addConstr(
        z_final <= (prob_terminate) * (terminate_upper * upsilon - alpha) + alpha
    )
    model.optimize()

    return get_probs(model, prob_vars, layers, states, _print)


def solve_layers_extern(L, lamb, _p2_goal, _p3_goal):
    global p2, p3, k
    p2, p3 = float128(_p2_goal), float128(_p3_goal)
    k = lamb * d
    uppers, lowers = compute_uppers(), compute_lowers()
    bad, badend, good, goodend = solve_layers(L, uppers, lowers, states, False)

    return bad, badend, good, goodend


if __name__ == "__main__":
    L = 700
    uppers, lowers = compute_uppers(), compute_lowers()

    # results = []
    # TIMEOUT = 60
    # for l in range(0, L + 1, 20):
    #     print(l)

    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(solve_layers, l, uppers, lowers, states, False)
            
    #         try:
    #             bad, badend, good, goodend = future.result(timeout=TIMEOUT)
    #         except concurrent.futures.TimeoutError:
    #             print(f"Skipping l={l} due to timeout")
    #             continue 

    #     terminate_lower = (k - d - 2) / (n * k)
        
    #     denominator = k + d * (1 + 2 * p2) - 2
        
    #     beta = (
    #         bad * (k - d - 2) ** 2 / (n1 * k * denominator)  # bad term
    #         + good * (k - d - 2) / denominator  # good term
    #     )
        
    #     alpha = badend / goodend if goodend != 0 else 0

    #     gamma = np.power(1 - terminate_lower, l) * (((k + 2 * p2 * d) / (n1 * k)) * (1 / beta) - alpha) + alpha
        
    #     results.append((l, gamma))  # Store (l, gamma) tuples

    # x_values, y_values = zip(*results)
    # plt.plot(x_values, y_values)
    # plt.xlabel("# of Layers")
    # plt.ylabel("Gamma")
    # plt.show()
    
    bad, badend, good, goodend = solve_layers(L, uppers, lowers, states, True)

    print(
        f"RESULTS: Bad - {bad},  BadEnd - {badend}, Good - {good}, GoodEnd - {goodend}"
    )

    print(badend/goodend)
