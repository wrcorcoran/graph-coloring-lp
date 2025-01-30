from itertools import chain, combinations_with_replacement, product, zip_longest
from pulp import *
from scipy.optimize import minimize
from levels import solve_layers
import matplotlib.pyplot as plt

# DISCLAIMER: THIS CODE ONLY WORKS FOR mstar = 3


NMAX = 7
mstar = 3
d = 5e2
n = 12 * d

layers = None
g_bad = None
g_good = None
g_goodend = None
num_layers = None

output_H = False

# k_goal = 1.832 * d
# p2_goal = 1/3
# c = (k_goal + 2 * p2_goal * d) / (k_goal - d - 2)
# gamma = ((6 * k_goal - d - 2) * (k_goal + 2 * p2_goal * d)) / (4 * (k_goal - d - 2) * (k_goal - d - 1))
# gamma = ((k_goal + (1 + 2 * p2_goal) * d - 2) * (k_goal + 2 * p2_goal * d)) / ((k_goal - d - 2) ** 2)
# gamma = (((n * k_goal) ** 2) * (k_goal + 2 * p2_goal * d)) / (((k_goal - d - 2) ** 2) * (n * k_goal - 2 * (1 + p2_goal) * d))
# print(c, gamma)
# c_gamma = 25.597784
# c_gamma = c * gamma
# print(c_gamma)
# c_gamma = 1/1000000000000
# c_gamma = 10000

p_vars = [
    LpVariable(f"p{i}", lowBound=1 if i == 1 else 0, upBound=1 / i)
    for i in range(1, NMAX + 1)
]
lambdas = {
    key: LpVariable(f"lambda_{key}", lowBound=1, upBound=2)
    for key in ["bad", "other", "good"]
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
    vecs = list(combinations_with_replacement(range(1, NMAX + 1), length))
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


# def case_to_lambda(case):
#     A, B, av, bv = case

#     if (A == 3 and B == 7 and av == (1, 1) and bv == (3, 3)) or (
#         A == 7 and B == 3 and av == (3, 3) and bv == (1, 1)
#     ):
#         return lambdas["bad"]

#     # if len(av) == 1:
#     #     print(case)

#     return lambdas["other"] if len(av) == 1 else lambdas["good"]


def case_to_lambda(case):
    A, B, av, bv = case

    is_good, is_bad = False, False

    # all unblocked:
    # dc = 1, av (1) bv (1)
    # dc = 2, av (1, 1) bv (1, 1)
    # if all(av[i] == 1 for i in range(len(av))) and all(bv[i] == 1 for i in range(len(bv))) and len(av) == len(bv):
    #     is_good = True

    # # >= singly blocked if
    # # if at any point in av or bv, one of these is 1
    # # and it has a nonzero difference
    # for i in range(len(av)):
    #     if abs(av[i] - bv[i]) >= 1:
    #         # print(A, B, av, bv)
    #         is_bad = True

    for i in range(len(av)):
        # there exists an unblocked config
        if av[i] == bv[i] == 1:
            is_good = True

        # there exists >= 1 singly blocked config
        if abs(av[i] - bv[i]) == 1 and min(av[i], bv[i]) == 1:
            is_bad = True

    if not (is_good or is_bad):
        # print("Other: ", A, B, av, bv)
        return lambdas["other"]

    # if is_good:
    #     # print("Good: ", A, B, av, bv)
    #     pass

    # if is_bad:
    #     print("Bad: ", A, B, av, bv)

    return lambdas["good"] if is_good else lambdas["bad"]


def setup(c_gamma, k_goal, p2_goal, p3_goal):
    prob = LpProblem("Chen_et_al_Variable-Length_Coupling", LpMinimize)
    lambda_obj = LpVariable("lambda_obj", lowBound=1, upBound=2)
    prob += lambda_obj
    prob += (
        lambda_obj - k_goal >= 0
    )  # i'm not sure if this is right, this is how they write it in their paper
    prob += lambda_obj - lambdas["other"] >= 0, "LP 4, Page 16, l_obj >= l_other"
    prob += lambda_obj - lambdas["good"] >= 0, "LP 4, Page 16, l_obj >= l_good"
    prob += (
        lambda_obj
        - lambdas["bad"] * (c_gamma) / (c_gamma + 1)
        - lambdas["good"] * 1 / (c_gamma + 1)
        >= 0,
        "LP 4, Page 16, c_gamma Mixed Coupling Final Constraints",
    )
    prob += p(2) - p2_goal <= 0
    prob += p(3) - p3_goal <= 0

    return prob


def flip_constraints():
    constraints = [p(1) == 1]

    for i in range(1, NMAX + 1):
        constraints.append(p(i) - p(i + 1) >= 0)
        constraints.append(p(i) * i <= 1)

    return constraints


min_memo = {}
min_consts = []
min_vars = []


def min_var(l):
    if len(l) == 4:
        a, A, b, B = l
        if a > b:
            a, A, b, B = b, B, a, A

        l = (a, A, b, B)
        if l in min_memo:
            return min_memo[l]

        if A >= B:
            min_memo[l] = p(b) - p(B)
        elif A >= NMAX + 1:
            min_memo[l] = p(b) - p(B)
        elif B >= NMAX + 1:
            min_memo[l] = min_var((b, a, B))
        else:
            new_var = LpVariable(f"min-{a}-{A}-{b}-{B}", lowBound=0, upBound=1)
            min_memo[l] = new_var
            min_consts.append(p(a) - p(A) - min_memo[l] >= 0)
            min_consts.append(p(b) - p(B) - min_memo[l] >= 0)

    else:
        b, a, A = l

        if l in min_memo:
            return min_memo[l]

        if a >= b:
            min_memo[l] = p(b)
        elif A >= NMAX + 1:
            min_memo[l] = p(a)
        else:
            new_var = LpVariable(f"min-{b}-{a}-{A}", lowBound=0, upBound=1)
            min_memo[l] = new_var
            min_consts.append(p(b) - min_memo[l] >= 0)
            min_consts.append(p(a) - p(A) - min_memo[l] >= 0)

    return min_memo[l]


def f(A, B, av, bv, i, imax, jmax):
    qi = p(av[i]) - p(A) if i == imax else p(av[i])
    qi_p = p(bv[i]) - p(B) if i == jmax else p(bv[i])

    # four cases (imax always 1)
    # 1. i = 0 & (imax = jmax)
    # 2. i = 0 & (imax != jmax)
    # 3. i = 1 & (imax = jmax)
    # 4. i = 1 & (imax != jmax)

    min_qi_qi_p = None
    if i == 0:
        if imax == jmax:
            min_qi_qi_p = min_var((av[i], A, bv[i], B))
        else:
            min_qi_qi_p = min_var((bv[i], av[i], A))
    else:  # i == 1
        if imax == jmax:
            min_qi_qi_p = p(max(av[i], bv[i]))
        else:
            min_qi_qi_p = min_var((av[i], bv[i], B))

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

    if output_H:
        print(f"{A}, {B}, {av}, {bv}, H - {(1 + value(H)) / length}")

    # only constrain the tightest
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
        i = 0
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


def solve(_k_goal, _p2_goal, _p3_goal, is_final=False):
    global g_bad, g_good, g_goodend
    pbad, pgood, pgoodend = solve_layers(num_layers, _k_goal, _p2_goal, _p3_goal)
    # pbad, pgood = solve_layers(10)
    g_bad, g_good, g_goodend = pbad, pgood, pgoodend
    k_goal, p2_goal = _k_goal * d, _p2_goal
    c = (k_goal + 2 * p2_goal * d) / (k_goal - d - 2)
    gamma = None
    if not layers:
        gamma = ((k_goal + (1 + 2 * p2_goal) * d - 2) * (k_goal + 2 * p2_goal * d)) / (
            (k_goal - d - 2) * (k_goal - d - 2)
        )
    else:
        gamma = ((k_goal + 2 * p2_goal * d) / (n * k_goal)) * (1 / (
            g_goodend
            + g_good * (k_goal - d - 2) / (k_goal + d * (1 + 2 * p2_goal) - 2)
            + (1 - g_good)
            * ((k_goal - d - 2)
            / (k_goal + d * (1 + 2 * p2_goal) - 2))
            * ((k_goal - d - 2)
            / (n * k_goal))
        ))
        # bad, good = g_bad, g_good
        # # bad, good = 0, 1
        # pbad, pgood = (bad) / (bad + good), (good) / (bad + good)
        # # bad_mult = (k_goal - d - 2) * (k_goal - d - 2) / (n * k_goal * (k_goal + d * (1 + 2 * p2_goal) - 2))
        # # good_mult = (k_goal - d - 2) / (k_goal + d * (1 + 2 * p2_goal) - 2)
        # # pbad, pgood = (good_mult) / (bad_mult + good_mult), (bad_mult) / (bad_mult + good_mult)

        # gamma = ((k_goal + 2 * p2_goal * d) / (n * k_goal)) * (
        #     1
        #     / (
        #         pbad
        #         * (k_goal - d - 2)
        #         * (k_goal - d - 2)
        #         / (n * k_goal * (k_goal + d * (1 + 2 * p2_goal) - 2))  # bad
        #         + (
        #             pgood * (k_goal - d - 2) / (k_goal + d * (1 + 2 * p2_goal) - 2)
        #         )  # good
        #     )
        # )

        # print(f"BAD: {pbad}, GOOD: {pgood}, Eval: {gamma}")
    # print(f"C: {c}, γ: {gamma}")
    c_gamma = c * gamma
    prob = setup(c_gamma, _k_goal, _p2_goal, _p3_goal)
    add_constraints(prob)

    # prob.writeLP("ChenLP.lp")
    # solver = pulp.PULP_CBC_CMD(msg=False)
    solver = GUROBI_CMD(msg=False, warmStart=True)
    prob.solve(solver)
    impt_vars = set(
        [
            "lambda_bad",
            "lambda_good",
            "lambda_obj",
            "lambda_other",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
        ]
    )
    # print(f'LP for ')
    if is_final:
        for v in prob.variables():
            if v.name in impt_vars:
                print(v.name, "=", v.varValue)
        # global output_H
        # output_H = True
        # hamming_constraints()
    val = prob.variablesDict()["lambda_obj"].varValue

    if prob.status != 1:
        return float("inf")
    del prob
    return val


def objective(params):
    param1, param2, param3 = params
    if param1 > 11 / 6 or param1 < 1.6:
        return float("inf")
    if param2 > 1 / 3 or param2 <= 0:
        return float("inf")
    if param3 > 1 / 4 or param3 <= 0:
        return float("inf")
    print(f"Trying: Lambda - {param1}, P2 - {param2}, P3 - {param3}")
    try:
        res = solve(param1, param2, param3)
        return res
    except Exception as e:
        return float("inf")


if __name__ == "__main__":
    # global g_bad, g_good, layers
    # 1.8322602
    layers = True
    # with layers
    if layers:
        results = []
        for i in range(70, 1001, 1):
            print(f"Testing {i}")
            num_layers = i
            initial_guess = [1.8242894, 0.32009043, 0.16004522]

            result = minimize(
                objective, initial_guess, method="Nelder-Mead", options={"maxiter": 50}
            )
            print("Optimized parameters:", result.x)
            print("Minimum value:", result.fun)
            final = tuple(result.x)
            if result.fun == float("inf"):
                continue
            solve(final[0], final[1], final[2], is_final=True)
            results.append((i, final[0]))

        x_values, y_values = zip(*results)
        plt.plot(x_values, y_values)
        plt.xlabel("# of Layers")
        plt.ylabel("Lambda")
        plt.show()

        # num_layers = int(d)
        # initial_guess = [1.8242894, 0.32009043, 0.16004522]

        # result = minimize(objective, initial_guess, method="Nelder-Mead", options={'maxiter': 50})
        # print("Optimized parameters:", result.x)
        # print("Minimum value:", result.fun)
        # final = tuple(result.x)
        # solve(final[0], final[1], final[2], is_final=True)

    else:
        # # without layers
        initial_guess = [1.8325937, 0.30994069, 0.16629646]
        result = minimize(objective, initial_guess, method="Nelder-Mead")
        print("Optimized parameters:", result.x)
        print("Minimum value:", result.fun)
        final = tuple(result.x)
        solve(final[0], final[1], final[2], is_final=True)
    # solve(initial_guess[0], initial_guess[1], True)
