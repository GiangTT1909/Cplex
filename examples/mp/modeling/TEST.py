import csv
import numpy as np
from collections import namedtuple
from array import *
from docplex.mp.model import Model
from docplex.util.environment import get_environment
import pandas as pd
import math

def column(matrix, i):
    return [row[i] for row in matrix]


population_temp = pd.read_excel(r"C:\Users\ACER\Downloads\Gs-master\gas\data\dataset1.xlsx")




TOTAL_CANDIDATE = 250



Skill = namedtuple("SKILL", ["name", "id"])
TOTAL_SKILL = 30

GROUP = [("Anne", 2),
         ("Anne", 2),
         ("Anne", 1)


         ]

Group = namedtuple("GROUP", ["name", "needed_member"])
TOTAL_GROUP = len(GROUP)



NEEDED_SKILL = [("Anne", 1, 1, 1,0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1),
                ("Anne", 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1),
                ("Anne", 1,1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,1, 1, 1, 1, 1, 1, 0, 1, 1, 1,1, 1, 1, 1, 0, 1, 1, 0)]



                

REQUIREMENT = [(1000, 0, 1000),
               (0, 1000, 1000),
               (0, 1000, 1000)

               ]
# listb = [x for x in lista]
# int("The original list : " + str(CANDIDATE_SKILL))

# using max() + list comprehension + zip()
# Maximum of each Column


def build_model(**kwargs):
    
    requirement = np.array(REQUIREMENT)
    requirement = requirement.astype(np.float)
    a = np.array(population_temp)
    r_data = a.transpose()
    a1 = np.array(r_data)

    r_data = r_data.astype(np.float)
    les = np.sort(r_data, axis=1)

    res = -np.sort(-r_data, axis=1)

    group_need = np.array(GROUP)
    group_need = np.delete(group_need, 0, 1)
    group_need = group_need.transpose()
    group_need = group_need.astype(np.int32)

    Weight = [0.4,0.3,0.3]
    # model
    mdl = Model("MCDM", **kwargs)
    candidate_skill = np.array(population_temp)

    a = np.array(NEEDED_SKILL)
    candidate_skill = np.array(population_temp)

    candidate_skill = candidate_skill.astype(np.float)

    needed_skill = np.array(NEEDED_SKILL)
    needed_skill = np.delete(needed_skill, 0, 1)
    needed_skill = needed_skill.astype(np.float)
    
    # decision variables
    x = mdl.binary_var_matrix(
        TOTAL_CANDIDATE, TOTAL_GROUP, lambda ij: "x_%d_%d" % (ij[0], ij[1]))
    
    # deep objective
    deep = {}
    for j in range(0, TOTAL_GROUP):
        deep_obj = 0
        for n in range(0, TOTAL_SKILL):
            for i in range(0, TOTAL_CANDIDATE):
                deep_obj +=  x[i, j]*candidate_skill[i][n]*needed_skill[j][n]
        deep[j] = deep_obj
    # wide objective
    wise = {}
    for j in range(0, TOTAL_GROUP):
        wise_obj = 0
        for n in range(0, TOTAL_SKILL):
            for i in range(0, TOTAL_CANDIDATE):

                wise_obj +=     x[i, j]*min(1, candidate_skill[i][n]*needed_skill[j][n])
        wise[j] = wise_obj
    max_point_deep = {}
    min_point_deep = {}

    for g in range(0, TOTAL_GROUP):
        a = 0
        b = 0
        for s in range(0, TOTAL_SKILL):
            for i in range(0, group_need[0][g]):
                a += needed_skill[g, s]*res[s][i]
                b += needed_skill[g, s]*les[s][i]
        max_point_deep[g] = a
        min_point_deep[g] = b

    max_point_wise = {}
    min_point_wise = {}
    
    for g in range(0, TOTAL_GROUP):
        a = 0
        b = 0
        for s in range(0, TOTAL_SKILL):
            for i in range(0, group_need[0][g]):
                a += min(1, needed_skill[g, s]*res[s][i])
                b += min(1, needed_skill[g, s]*les[s][i])
        max_point_wise[g] = a
        min_point_wise[g] = b
    print(max_point_wise)
    print(min_point_wise)
    distance_deep = 0
    distance_wise = 0
    print(max_point_deep)
    print(min_point_deep)
    
    # distance_deep =mdl.sum(Weight[g]*(((max_point_deep[g] -
    #                                    deep[g])/(max_point_deep[g]-min_point_deep[g]))**2) for g in range(0, TOTAL_GROUP))
    # distance_wise = mdl.sum(Weight[g]*(((max_point_wise[g] -
    #                                    wise[g])/(max_point_wise[g]-min_point_wise[g]))**2) for g in range(0, TOTAL_GROUP))
    
    distance_deep += mdl.sum(Weight[g]*(((max_point_deep[g] -
                                       deep[g])/(max_point_deep[g]-min_point_deep[g]))**2) for g in range(0, TOTAL_GROUP))
    distance_wise +=  mdl.sum(Weight[g]*(((max_point_wise[g] -
                                       wise[g])/(max_point_wise[g]-min_point_wise[g]))**2) for g in range(0, TOTAL_GROUP))
    # # mdl.deep = mdl.sum(x[i, 1]*candidate_skill[i][n]*needed_skill[1][n]
    # #                    for i in range(0, TOTAL_CANDIDATE) for n in range(0, TOTAL_SKILL))
    distance=distance_wise+distance_wise
    mdl.deep1 = mdl.sum(Weight[g]*(((max_point_deep[g] -
                                       deep[g])/(max_point_deep[g]-min_point_deep[g]))**2) for g in range(0, TOTAL_GROUP))
    mdl.wise = mdl.sum(Weight[g]*(((max_point_wise[g] -
                                       wise[g])/(max_point_wise[g]-min_point_wise[g]))**2) for g in range(0, TOTAL_GROUP))
    mdl.add_kpi(mdl.deep1, "Total salary cost")
    mdl.add_kpi(mdl.wise, "Totalt")

    mdl.add_constraint(mdl.sum(x[i, j] for j in range(
        0, TOTAL_GROUP) for i in range(0, TOTAL_CANDIDATE)) == 5)
    mdl.add_constraints(mdl.sum(x[i, j] for j in range(
        0, TOTAL_GROUP)) <= 1 for i in range(0, TOTAL_CANDIDATE))
    mdl.add_constraints(mdl.sum(x[i, j] for i in range(
        0, TOTAL_CANDIDATE)) == group_need[0][j] for j in range(0, TOTAL_GROUP))
    # mdl.add_constraints(mdl.sum(x[i,g]*candidate_skill[i][s] for i in range(0,TOTAL_CANDIDATE)) >= requirement[g][s] for g in range(0,TOTAL_GROUP) for s in range(0,TOTAL_SKILL))
    # mdl.minimize_static_lex([distance_deep,distance_wise])
    mdl.minimize(distance)
    return mdl


def checkContraint1(solution):
    solution = np.array(solution)
    solution = solution.astype(np.int32)
    for i in range(0, TOTAL_GROUP):
        sum = 0
        for j in range(0, TOTAL_CANDIDATE):
            sum += solution[j][i]
        if(sum > 1):
            return False
    return True


def checkContraint2(solution):
    solution = np.array(solution)
    solution = solution.astype(np.int32)
    sum = 0
    for i in range(0, TOTAL_GROUP):

        for j in range(0, TOTAL_CANDIDATE):
            sum += solution[j][i]
    if(sum != 6):
        return False
    return True


if __name__ == '__main__':
    model = build_model()
    s = model.solve(log_output=True)
    # # Solve the model and print solution
    # model.solve()

    # # Save the CPLEX solution as "solution.json" program output
    if s:
        model.report()
        model.print_solution(log_output=True)

        # Save the CPLEX solution as "solution.json" program output
        with get_environment().get_output_stream("solution.json") as fp:
            model.solution.export(fp, "json")
    else:
        print("Problem has no solution")
