from itertools import chain, combinations_with_replacement, product
from pulp import *

NMAX = 7
mstar = 3
gamma = 25.597784

p_vars = [
    LpVariable(f"p{i}", lowBound=1 if i == 1 else 0, upBound=1)
    for i in range(1, NMAX + 1)
]
lambdas = {
    key: LpVariable(f"lambda_{key}", lowBound=1, upBound=2)
    for key in ["bad", "sing", "good"]
}
lambda_obj = LpVariable("lambda_obj", lowBound=1, upBound=2)


# from chen et al code
def p(i):
    if i in range(1, NMAX + 1):
        return p_vars[i - 1]
    else:
        return 0


# picks out amax and aimax
def maxes(v):
    mx, mxi = 0, 0
    for i, x in enumerate(v):
        if x > mx:
            mxi = i
            mx = x
    return mx, mxi


def build_vecs(length):
    vecs = list(combinations_with_replacement(range(0, NMAX + 1), length))
    return [x for x in vecs if sum(x) != 0]


def build_pair_vecs(length):
    vecs = [tuple(sorted(t, reverse=True)) for t in build_vecs(length)]
    pairs = combinations_with_replacement(vecs, 2)

    # note, this will break if mstar > 3, should reimplement
    if length > 1:
        swapped = [(t[1], t[0]) for t in vecs]
        swapped_pairs = product(vecs, swapped)
        return chain(pairs, swapped_pairs)

    return pairs


def case_to_lambda(case):
    A, B, av, bv = case

    if (A == 3 and B == 7 and av == (1, 1) and bv == (3, 3)) or (
        A == 7 and B == 3 and av == (3, 3) and bv == (1, 1)
    ):
        return lambdas["bad"]

    return lambdas["sing"] if len(av) == 1 else lambdas["good"]


def setup():
    prob = LpProblem("Chen_et_al_Variable-Length_Coupling", LpMinimize)
    lambda_obj = LpVariable("lambda_obj", lowBound=1, upBound=2)
    prob += lambda_obj, "Objective: Minize LambdaObj"
    prob += lambda_obj - lambdas["sing"] >= 0, "LP 4, Page 16, l_sing >= l_sing"
    prob += lambda_obj - lambdas["good"] >= 0, "LP 4, Page 16, l_sing >= l_good"
    prob += (
        lambda_obj
        - lambdas["bad"] * (gamma) / (gamma + 1)
        - lambdas["good"] * 1 / (gamma + 1)
        >= 0,
        "LP 4, Page 16, Gamma Mixed Coupling Final Constraints",
    )

    return prob


def flip_constraints():
    constraints = [p(1) == 1]

    for i in range(1, NMAX + 1):
        constraints.append(p(i) - p(i + 1) >= 0)
        constraints.append(p(i) * i <= 1)

    return constraints


min_memo = {}
min_consts = []


def min_var(expr1, expr2, name):
    if name in min_memo:
        return min_memo[name]
    min_val = pulp.LpVariable(name, upBound=1)
    min_consts.append(min_val <= expr1)
    min_consts.append(min_val <= expr2)
    min_memo[name] = min_val
    return min_val


def f(A, B, av, bv, i, imax, jmax):
    qi = p(av[i]) - p(A) if i == imax else p(av[i])
    qi_p = p(bv[i]) - p(B) if i == jmax else p(bv[i])

    min_qi_qi_p = min_var(qi, qi_p, f"min_qi_qi_p_{str(av)}_{str(av)}")

    return av[i] * qi + bv[i] * qi_p - min_qi_qi_p


def constraints_11(av, bv):
    constraints = []
    amax, aimax = maxes(av)
    bmax, bimax = maxes(bv)

    length = len(av)

    A_lower = amax + 1
    B_lower = bmax + 1
    A_upper = sum(av) + 1
    B_upper = sum(bv) + 1

    # calculate f, using (A - a max - 1)PA + (B - b max -1)PB - Sum_i (f_i)
    for A in range(A_lower, A_upper + 1):
        for B in range(B_lower, B_upper + 1):
            H = (
                (A - amax - 1) * p(A)
                + (B - bmax - 1) * p(B)
                + sum(f(A, B, av, bv, i, aimax, bimax) for i in range(0, length))
            )
            lamb = case_to_lambda((A, B, av, bv))
            constraints.append(1 + H - (lamb * length) <= 0)

    return constraints


def constraints_12(bv):
    length = len(bv)
    B = sum(bv)

    return (
        1 - lambdas["good"] * length + (B - bv[-1]) * p(B) + sum(b * p(b) for b in bv)
        <= 0
    )


def constraints_14():
    constraints = []
    x, y = LpVariable("x"), LpVariable("y")
    for A in range(0, NMAX + 2):
        constraints.append(x - (A - 2) * p(A) >= 0)
    for a in range(0, NMAX + 1):
        for b in range(a, NMAX + 1):
            constraints.append(a * p(a) + (b - 1) * p(b) - y <= 0)
    constraints.append(2 * x + mstar * y + 1 - lambdas["good"] * mstar <= 0)
    return constraints


def hamming_constraints():
    constraints = []

    # constraint 12
    for length in range(2, mstar):
        b_vecs = build_vecs(length)
        for bv in b_vecs:
            constraints.append(constraints_12(bv))

    # constraint 11
    for length in range(1, mstar):
        pairs = build_pair_vecs(length)
        for av, bv in pairs:
            constraints.extend(constraints_11(av, bv))

    constraints.extend(min_consts)

    return constraints


def add_constraints(prob):
    constraints = []
    constraints.extend(flip_constraints())
    constraints.extend(hamming_constraints())
    constraints.extend(constraints_14())

    for c in constraints:
        if c is None:
            continue
        prob += c


if __name__ == "__main__":
    prob = setup()
    add_constraints(prob)

    # print("Problem:\n")
    # print(prob)
    # print("\n" * 5)

    prob.writeLP("ChenLP.lp")
    prob.solve()
    for v in prob.variables():
        print(v.name, "=", v.varValue)