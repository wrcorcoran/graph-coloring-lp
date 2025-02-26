from numpy import float128
from pulp import *
import matplotlib.pyplot as plt
from gurobipy import read

# print(listSolvers())
solver = GUROBI_CMD(msg=False, warmStart=True)
# solver = PULP_CBC_CMD(msg=False, options=["feasibilitypump"])

# DISCLAIMER: CODE IS FUNCTIONALLY IMPLEMENTED
NMAX = 7
mstar = 3
dc = mstar - 1
d = 5e2
lamb = 1.8287944
# n = 1e5
n = d * 12
k = lamb * d
# p2 = 0.30994691
p2 = 0.31211749
# p3 = 0.16629646
p3 = 0.16439719
n1 = 1e10


k = float128(k)
d = float128(d)
n = float128(n)
p2 = float128(p2)
p3 = float128(p3)
dc = float128(dc)

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
        "OTHER": 1 - ((dc * (3 * p3 * (d - 1) + 2 * k - 6) + k + 2 * p2 * d) / (n * k)),
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

states = ["GOOD", "BAD", "OTHER", "BADEND", "OTHEREND", "GOODEND"]


def __name(layer, state):
    return f"{state}_{layer}"


prob_vars = {
    __name(0, name): LpVariable(__name(0, name), lowBound=0, upBound=1)
    for name in states
}


def initial_constraints(prob):
    prob += sum(prob_vars[__name(0, state)] for state in states) == 1
    prob += prob_vars[__name(0, "BADEND")] == 0
    prob += prob_vars[__name(0, "GOODEND")] == 0
    prob += prob_vars[__name(0, "OTHEREND")] == 0
    prob += prob_vars[__name(0, "BAD")] == 1


def add_layers(prob, l):
    bad_lower = sum(lowers[s]["BAD"] * prob_vars[__name(l - 1, s)] for s in states)
    bad_upper = sum(uppers[s]["BAD"] * prob_vars[__name(l - 1, s)] for s in states)

    good_lower = sum(lowers[s]["GOOD"] * prob_vars[__name(l - 1, s)] for s in states)
    good_upper = sum(uppers[s]["GOOD"] * prob_vars[__name(l - 1, s)] for s in states)

    other_lower = sum(lowers[s]["OTHER"] * prob_vars[__name(l - 1, s)] for s in states)
    other_upper = sum(uppers[s]["OTHER"] * prob_vars[__name(l - 1, s)] for s in states)

    bad_end_lower = sum(
        lowers[s]["BADEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    bad_end_upper = sum(
        uppers[s]["BADEND"] * prob_vars[__name(l - 1, s)] for s in states
    )

    good_end_lower = sum(
        lowers[s]["GOODEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    good_end_upper = sum(
        uppers[s]["GOODEND"] * prob_vars[__name(l - 1, s)] for s in states
    )

    other_end_lower = sum(
        lowers[s]["OTHEREND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    other_end_upper = sum(
        uppers[s]["OTHEREND"] * prob_vars[__name(l - 1, s)] for s in states
    )

    p_bad = LpVariable(__name(l, "BAD"), lowBound=0, upBound=1)
    p_good = LpVariable(__name(l, "GOOD"), lowBound=0, upBound=1)
    p_other = LpVariable(__name(l, "OTHER"), lowBound=0, upBound=1)
    p_bad_end = LpVariable(__name(l, "BADEND"), lowBound=0, upBound=1)
    p_good_end = LpVariable(__name(l, "GOODEND"), lowBound=0, upBound=1)
    p_other_end = LpVariable(__name(l, "OTHEREND"), lowBound=0, upBound=1)

    prob += p_bad + p_good + p_other + p_bad_end + p_good_end + p_other_end == 1

    prob += p_bad <= bad_upper
    prob += p_bad >= bad_lower

    prob += p_good <= good_upper
    prob += p_good >= good_lower

    prob += p_other <= other_upper
    prob += p_other >= other_lower

    prob += p_bad_end <= bad_end_upper
    prob += p_bad_end >= bad_end_lower

    prob += p_good_end <= good_end_upper
    prob += p_good_end >= good_end_lower

    prob += p_other_end <= other_end_upper
    prob += p_other_end >= other_end_lower

    prob_vars[__name(l, "BAD")] = p_bad
    prob_vars[__name(l, "GOOD")] = p_good
    prob_vars[__name(l, "OTHER")] = p_other
    prob_vars[__name(l, "BADEND")] = p_bad_end
    prob_vars[__name(l, "GOODEND")] = p_good_end
    prob_vars[__name(l, "OTHEREND")] = p_other_end


def get_probs(prob, layers, _print=False):
    if _print:
        impt_vars = set(
            [
                "bad_final",
                "z_final",
                *[__name(0, s) for s in states],
                *[__name(layers, s) for s in states],
            ]
        )
        # print(f'LP for ')
        for v in prob.variables():
            if v.name in impt_vars:
                print(v.name, "=", v.varValue)
    bad, good, goodend = (
        prob.variablesDict()[__name(layers, "BAD")].varValue,
        prob.variablesDict()[__name(layers, "GOOD")].varValue,
        prob.variablesDict()[__name(layers, "GOODEND")].varValue,
    )

    if prob.status != 1:
        print(LpStatus[prob.status])
        return float("inf"), float("inf")

    return bad, good, goodend


def solve_layers_all(layers):
    prob = LpProblem("LayerByLayer", LpMaximize)
    z_final = LpVariable("z_final", lowBound=0, upBound=1)
    prob += z_final

    initial_constraints(prob)

    for l in range(1, layers + 1):
        add_layers(prob, l)

    # z = p(bad) / p(good), z * p(good) <= p(bad)
    # prob += z_final >= prob_vars[__name(layers, "BAD")] * (k - d - 2) * (k - d - 2) / (
    #     n1 * k * (k + d * (1 + 2 * p2) - 2)
    # )  # bad
    # +(
    #     prob_vars[__name(layers, "GOOD")] * (k - d - 2) / (k + d * (1 + 2 * p2) - 2)
    # )  # good
    # +prob_vars[__name(layers, "GOODEND")]  # good end
    # prob += 1 / prob_vars[__name(0, "BAD")] >= 0

    prob += z_final >= prob_vars[__name(layers, "BAD")]

    prob.solve(solver)

    return get_probs(prob, layers, True)


def solve_layers(layers, _lamb=1.8287944, _p2=0.31211749, _p3=0.16439719):
    global p2, p3, lamb
    p2, p3, lamb = _p2, _p3, _lamb
    bad, good, goodend = solve_layers_all(layers)
    return bad, good, goodend


if __name__ == "__main__":
    print(uppers)
    print()
    print(lowers)
    print()

    bad, good, goodend = solve_layers(10)

    print(bad, good, goodend)

    print(f"RESULTS: Bad - {bad}, Good - {good}, GoodEnd - {goodend}")
