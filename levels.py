from itertools import chain, combinations_with_replacement, product
from numpy import float128
from pulp import *
from scipy.optimize import minimize

# DISCLAIMER: CODE IS FUNCTIONALLY IMPLEMENTED
NMAX = 7
mstar = 3
d = 1e2
lamb = 1.8325929
n = 1e5
k = lamb * d 
p2 = 0.30994691
p3 = 0.16629646


k = float128(k)
d = float128(d)
n = float128(n)
p2 = float128(p2)
p3 = float128(p3)

uppers = {
    "BAD": {
        "BAD": float128(1) - ((k - d - float128(2)) / (n * k)),
        "OTHER": (((d - float128(2)) * (float128(1) + p2)) / (d * n)) / ((n * k) / (d * n)),
        "GOOD": (k + (d * (p2 - float128(1))) - (p2 + float128(2))) / (n * k),
    },
    "OTHER": {
        "BAD": (float128(2) * d * (k - d - float128(2) + (p2 * (d - float128(1))))) / (n * k),
        "OTHER": float128(1),
        "GOOD": (float128(3) * d * p3) / (n * k),
    },
    "GOOD": {
        "BAD": (float128(2) * d * (float128(1) + p2)) / (n * k),
        "OTHER": (d * (float128(3) * p3 + float128(2) * p2)) / (n * k),
        "GOOD": float128(1),
    },
}
lowers = {
    "BAD": {
        "BAD": float128(1) - ((k + p2 * ((float128(2) * d) - float128(3)) - float128(4)) / (n * k)),
        "OTHER": float128(0),
        "GOOD": (k - d - float128(2)) / (n * k),
    },
    "OTHER": {
        "BAD": float128(0),
        "OTHER": float128(1) - ((float128(2) * d) * (k - d - float128(2) + p2 * (d - float128(1))) * (float128(3) / float128(2) * p3) / (n * k)),
        "GOOD": float128(0),
    },
    "GOOD": {
        "BAD": float128(0),
        "OTHER": float128(0),
        "GOOD": float128(1) - ((float128(2) * d * (float128(1) + p2 + float128(3) / float128(2) * p3)) / (n * k)),
    },
}

states = ["GOOD", "BAD", "OTHER"]

def __name(layer, state):
     return f"{layer}_{state}"

prob_vars = {__name(0, name): LpVariable(__name(0, name), lowBound=0, upBound=1) for name in states}

def initial_constraints(prob):
    prob += prob_vars[__name(0, "GOOD")] + prob_vars[__name(0, "BAD")] + prob_vars[__name(0, "OTHER")] == 1


def add_layers(prob, l):
    bad_upper = uppers["BAD"]["BAD"] * prob_vars[__name(l - 1, "BAD")] + uppers["GOOD"]["BAD"] * prob_vars[__name(l - 1, "GOOD")] + uppers["OTHER"]["BAD"] * prob_vars[__name(l - 1, "OTHER")]
    good_upper = uppers["BAD"]["GOOD"] * prob_vars[__name(l - 1, "BAD")] + uppers["GOOD"]["GOOD"] * prob_vars[__name(l - 1, "GOOD")] + uppers["OTHER"]["GOOD"] * prob_vars[__name(l - 1, "OTHER")]
    other_upper = uppers["BAD"]["OTHER"] * prob_vars[__name(l - 1, "BAD")] + uppers["GOOD"]["OTHER"] * prob_vars[__name(l - 1, "GOOD")] + uppers["OTHER"]["OTHER"] * prob_vars[__name(l - 1, "OTHER")]

    bad_lower = lowers["BAD"]["BAD"] * prob_vars[__name(l - 1, "BAD")] + lowers["GOOD"]["BAD"] * prob_vars[__name(l - 1, "GOOD")] + lowers["OTHER"]["BAD"] * prob_vars[__name(l - 1, "OTHER")]
    good_lower = lowers["BAD"]["GOOD"] * prob_vars[__name(l - 1, "BAD")] + lowers["GOOD"]["GOOD"] * prob_vars[__name(l - 1, "GOOD")] + lowers["OTHER"]["GOOD"] * prob_vars[__name(l - 1, "OTHER")]
    other_lower = lowers["BAD"]["OTHER"] * prob_vars[__name(l - 1, "BAD")] + lowers["GOOD"]["OTHER"] * prob_vars[__name(l - 1, "GOOD")] + lowers["OTHER"]["OTHER"] * prob_vars[__name(l - 1, "OTHER")]

    p_bad = LpVariable(__name(l, "BAD"), lowBound=0, upBound=1)
    p_good = LpVariable(__name(l, "GOOD"), lowBound=0, upBound=1)
    p_other = LpVariable(__name(l, "OTHER"), lowBound=0, upBound=1)

    prob += p_bad + p_good + p_other == 1
    prob += p_bad <= bad_upper
    prob += p_bad >= bad_lower
    prob += p_good <= good_upper
    prob += p_good >= good_lower
    prob += p_other <= other_upper
    prob += p_other >= other_lower

    prob_vars[__name(l, "BAD")] = p_bad
    prob_vars[__name(l, "GOOD")] = p_good
    prob_vars[__name(l, "OTHER")] = p_other


def solve(layers):
    prob = LpProblem("LayerByLayer", LpMaximize)
    bad_final = LpVariable("bad_final", lowBound=0, upBound=1)
    prob += bad_final, "Objective: Maximize LambdaObj"

    initial_constraints(prob)

    for l in range(1, layers + 1):
        add_layers(prob, l)

    prob += bad_final <= prob_vars[__name(layers, "BAD")]

    prob.writeLP("levels.lp")
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    impt_vars = set([__name(0, "GOOD"), __name(0, "BAD"), __name(0, "OTHER"), __name(layers, "GOOD"), __name(layers, "BAD"), __name(layers, "OTHER"), "bad_final"])
    # print(f'LP for ')
    for v in prob.variables():
        if v.name in impt_vars:
            print(v.name, "=", v.varValue)
    val = prob.variablesDict()['bad_final'].varValue

    if prob.status != 1: return float('inf')
    del prob
    return val

if __name__ == "__main__":
    print(uppers)
    print()
    print(lowers)

    solve(int(1e3))
