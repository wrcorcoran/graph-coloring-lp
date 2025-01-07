from numpy import float128
from pulp import *

solver = pulp.PULP_CBC_CMD(msg=False)

# DISCLAIMER: CODE IS FUNCTIONALLY IMPLEMENTED
NMAX = 7
mstar = 3
d = 1e3
lamb = 1.8287944
n = 1e7
k = lamb * d
# p2 = 0.30994691
p2 = 0.31211749
# p3 = 0.16629646
p3 = 0.16439719


k = float128(k)
d = float128(d)
n = float128(n)
p2 = float128(p2)
p3 = float128(p3)

uppers = {
    "BAD": {
        "BAD": 1 - 2 * ((k - d - 2) / (n * k)),
        "OTHER": (((d - 2) * (1 + p2))) / ((n * k)),
        "GOOD": (k + (d * (p2 - 1)) - (p2 + 2)) / (n * k),
        "BADEND": (k + 2 * p2 * d) / (n * k),
        "OTHEREND": 0,
        "GOODEND": 0,
    },
    "OTHER": {
        "BAD": 2 * d * (k - d - 2 + (p2 * (d - 1))) / (n * k),
        "OTHER": 1 - (k - d - 2) / (n * k),
        "GOOD": (3 * d * p3) / (n * k),
        "BADEND": 0,
        "OTHEREND": (k + 2 * p2 * d) / (n * k),
        "GOODEND": 0,
    },
    "GOOD": {
        "BAD": (2 * d * (1 + p2)) / (n * k),
        "OTHER": (d * (3 * p3 + 2 * p2)) / (n * k),
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
        "BAD": 1 - ((2 * k + p2 * (4 * d - 3) - 4) / (n * k)),
        "OTHER": 0,
        "GOOD": (k - d - 2) / (n * k),
        "BADEND": (k - d - 2) / (n * k),
        "OTHEREND": 0,
        "GOODEND": 0,
    },
    "OTHER": {
        "BAD": 0,
        "OTHER": 1 - ((k + 2 * d * (k + d * (p2 - 1)) + (3 / 2) * p3) / (n * k)),
        "GOOD": 0,
        "BADEND": 0,
        "OTHEREND": (k - d - 2) / (n * k),
        "GOODEND": 0,
    },
    "GOOD": {
        "BAD": 0,
        "OTHER": 0,
        "GOOD": 1 - ((k + 2 * d * (1 + 2 * p2 + (3 / 2) * p3))) / (n * k),
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
    return f"{layer}_{state}"


prob_vars = {
    __name(0, name): LpVariable(__name(0, name), lowBound=0, upBound=1)
    for name in states
}


def initial_constraints(prob):
    prob += sum(prob_vars[__name(0, state)] for state in states) == 1


def add_layers(prob, l):
    bad_upper = sum(uppers[s]["BAD"] * prob_vars[__name(l - 1, s)] for s in states)
    good_upper = sum(uppers[s]["GOOD"] * prob_vars[__name(l - 1, s)] for s in states)
    other_upper = sum(uppers[s]["OTHER"] * prob_vars[__name(l - 1, s)] for s in states)
    bad_end_upper = sum(
        uppers[s]["BADEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    good_end_upper = sum(
        uppers[s]["GOODEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    other_end_upper = sum(
        uppers[s]["OTHEREND"] * prob_vars[__name(l - 1, s)] for s in states
    )

    bad_lower = sum(lowers[s]["BAD"] * prob_vars[__name(l - 1, s)] for s in states)
    good_lower = sum(lowers[s]["GOOD"] * prob_vars[__name(l - 1, s)] for s in states)
    other_lower = sum(lowers[s]["OTHER"] * prob_vars[__name(l - 1, s)] for s in states)
    bad_end_lower = sum(
        lowers[s]["BADEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    good_end_lower = sum(
        lowers[s]["GOODEND"] * prob_vars[__name(l - 1, s)] for s in states
    )
    other_end_lower = sum(
        lowers[s]["OTHEREND"] * prob_vars[__name(l - 1, s)] for s in states
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
                *[__name(0, s) for s in states],
                *[__name(layers, s) for s in states],
            ]
        )
        # print(f'LP for ')
        for v in prob.variables():
            if v.name in impt_vars:
                print(v.name, "=", v.varValue)
    bad, good = (
        prob.variablesDict()[__name(layers, "BAD")].varValue,
        prob.variablesDict()[__name(layers, "GOOD")].varValue,
    )

    if prob.status != 1:
        print(LpStatus[prob.status])
        return float("inf"), float("inf")

    return bad, good


def solve_layers_bad(layers):
    prob = LpProblem("LayerByLayer", LpMaximize)
    z_final = LpVariable("z_final", lowBound=0, upBound=1)
    prob += z_final, "Objective: Maximize z_final"

    initial_constraints(prob)

    for l in range(1, layers + 1):
        add_layers(prob, l)

    # z = p(bad) / p(good), z * p(good) <= p(bad)
    prob += z_final <= prob_vars[__name(layers, "BAD")]

    prob.writeLP("levels.lp")
    prob.solve(solver)

    return get_probs(prob, layers, _print=False)


def solve_layers_good(layers, bad):
    prob = LpProblem("LayerByLayer", LpMinimize)
    z_final = LpVariable("z_final", lowBound=0, upBound=1)
    prob += z_final, "Objective: Maximize z_final"

    initial_constraints(prob)

    for l in range(1, layers + 1):
        add_layers(prob, l)

    # z = p(bad) / p(good), z * p(good) <= p(bad)
    prob += prob_vars[__name(layers, "BAD")] == bad
    prob += z_final >= prob_vars[__name(layers, "GOOD")]

    prob.solve(solver)

    return get_probs(prob, layers, _print=True)


def solve_layers(layers):
    bad, _ = solve_layers_bad(layers)
    bad, good = solve_layers_good(layers, bad)
    return bad, good


if __name__ == "__main__":
    print(uppers)
    print()
    print(lowers)
    print()

    bad, good = solve_layers(int(1e3))
    print(bad, good)

    print(f"RESULTS: Bad - {bad}, Good - {good}")
