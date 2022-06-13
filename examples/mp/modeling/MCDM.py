import numpy as np
from collections import namedtuple

from docplex.mp.model import Model
from docplex.util.environment import get_environment

def column(matrix, i):
    return [row[i] for row in matrix]
CANDIDATE = [   ("Anne",1),
          ("Bethanie",2),
          ("Betsy",3),
          ("Cathy",4),
          ("Cecilia",5),
          ("Chris",6),
          ("Cindy",7),
          ("David",8),
          ("Debbie",9),
          ("Dee",10),
          ("Gloria",11),
          ("Isabelle",12),
          ("Jane",13),
          ("Janelle",14),
          ("Janice",15),
          ("Jemma",16),
          ("Joan",17),
          ("Joyce",18),
          ("Jude",19),
          ("Julie",20),
          ("Juliet",21),
          ("Kate",22),
          ("Nancy",23),
          ("Nathalie",24),
          ("Nicole",25),
          ("Patricia",26),
          ("Patrick",27),
          ("Roberta",28),
          ("Suzanne",29),
          ("Vickie",30),
          ("Wendie",31),
          ("Zoe",32)

]
Candidate = namedtuple("CANDIDATE",["name","id"])
TOTAL_CANDIDATE = len(CANDIDATE)

SKILL = [   ("Anne",1),
          ("Bethanie",2),
          ("Betsy",3),
          ("Cathy",4),
          ("Cecilia",5),
          ("Chris",6),
          ("Cindy",7),
          ("David",8),
          ("Debbie",9)                 
]

Skill = namedtuple("SKILL",["name","id"])
TOTAL_SKILL = len(SKILL)

GROUP = [   ("Anne",1),
          ("Bethanie",2),
          ("Betsy",2),
          ("Cathy",1)
]

Group = namedtuple("GROUP",["name","needed_member"])
TOTAL_GROUP = len(GROUP)

CANDIDATE_SKILL = [("Anne",0,2,5,4,2,5,4,4,2),
          ("Bethanie",1,2,1,4,2,5,4,4,2),
          ("Betsy",2,3,5,4,2,1,4,4,2),
          ("Cathy",1,2,5,4,2,2,4,4,2),
          ("Cecilia",1,4,2,5,2,1,2,4,2),
          ("Chris",5,2,1,1,2,5,4,4,2),
          ("Cindy",1,2,3,1,2,5,4,4,2),
          ("David",1,1,2,5,2,5,4,4,2),
          ("Debbie",2,1,3,3,2,5,4,4,2),
          ("Dee",1,1,2,3,2,5,4,4,2),
          ("Gloria",2,3,3,3,2,5,4,4,2),
          ("Isabelle",2,1,4,5,2,5,4,4,2),
          ("Jane",1,5,1,4,2,5,4,4,2),
          ("Janelle",3,1,5,1,2,5,4,4,2),
          ("Janice",1,2,3,5,2,1,4,4,2),
          ("Jemma",1,2,4,2,2,2,4,4,2),
          ("Joan",1,2,4,2,2,5,4,4,2),
          ("Joyce",5,1,1,2,2,5,4,4,2),
          ("Jude",2,3,4,2,2,5,4,4,2),
          ("Julie",1,2,3,2,2,5,4,4,2),
          ("Juliet",1,1,3,5,2,5,4,4,2),
          ("Kate",1,2,4,2,2,5,4,4,2),
          ("Nancy",1,2,1,5,2,5,4,4,2),
          ("Nathalie",2,3,2,4,2,5,4,4,2),
          ("Nicole",2,4,1,2,2,5,4,4,2),
          ("Patricia",2,4,1,2,2,1,4,4,2),
          ("Patrick",2,3,2,4,2,2,4,4,2),
          ("Roberta",1,2,5,1,3,4,1,2,4),
          ("Suzanne",1,2,4,1,2,5,4,1,2),
          ("Vickie",1,2,1,1,2,5,3,4,2),
          ("Wendie",1,2,2,1,2,5,2,3,2),
          ("Zoe",1,2,1,1,2,5,4,4,2)
]

NEEDED_SKILL = [("Anne",1,0,1,1,0,0,1,1,1),
          ("Bethanie",0,1,1,0,1,0,0,0,0),
          ("Betsy",1,0,1,0,1,0,1,1,0),
          ("Cathy",0,1,0,0,0,0,0,0,1)
]

REQUIREMENT = [(4,0,4,4,0,0,5,5,4),
          (0,7,8,0,9,0,0,0,0),
          (8,0,9,0,10,0,8,8,0),
          (0,4,0,0,0,0,0,0,4)
]
# listb = [x for x in lista]
# int("The original list : " + str(CANDIDATE_SKILL))
  
# using max() + list comprehension + zip()
# Maximum of each Column
def build_model(**kwargs):
    requirement = np.array(REQUIREMENT)
    requirement = requirement.astype(np.float)
    a = np.array(CANDIDATE_SKILL)
    SKILL_CANDIDATE = a.transpose()
    a1 = np.array(SKILL_CANDIDATE)
    skill_candidate = np.delete(a1,0,0)    
    skill_candidate = skill_candidate.astype(np.float)
    les = np.sort(skill_candidate, axis = 1)
    
    
    res =-np.sort(-skill_candidate, axis = 1)
    
    group_need = np.array(GROUP)
    group_need = np.delete(group_need,0,1)
    group_need = group_need.transpose()   
    group_need=group_need.astype(np.int32) 
    
    Weight = [0.2,0.4,0.2,0.4]
    #model
    mdl = Model("MCDM", **kwargs)
    candidate_skill = np.array(CANDIDATE_SKILL)
    
    a = np.array(NEEDED_SKILL)
    candidate_skill = np.array(CANDIDATE_SKILL)
    candidate_skill = np.delete(candidate_skill,0,1)
    candidate_skill =candidate_skill.astype(np.float)  
    
    needed_skill = np.array(NEEDED_SKILL)
    needed_skill = np.delete(needed_skill,0,1)
    needed_skill = needed_skill.astype(np.float)
    
    # decision variables
    x = mdl.binary_var_matrix(TOTAL_CANDIDATE,TOTAL_GROUP,lambda ij: "x_%d_%d" %(ij[0], ij[1]))
    #deep objective
    deep = {}
    for j in range(0,TOTAL_GROUP):
        deep_obj = 0
        for n in range(0,TOTAL_SKILL):
            for i in range(0,TOTAL_CANDIDATE):         
               deep_obj =  mdl.sum(x[i,j]*candidate_skill[i][n]*needed_skill[j][n])
        deep[j]=deep_obj
    #wide objective
    wise = {}
    for j in range(0,TOTAL_GROUP):
        wise_obj = 0
        for n in range(0,TOTAL_SKILL):
            for i in range(0,TOTAL_CANDIDATE): 
               
               wise_obj =  mdl.sum(x[i,j]*min(1,candidate_skill[i][n]*needed_skill[j][n]))
        wise[j]=wise_obj
    max_point_deep = {}
    min_point_deep = {}
    
    for g in range (0,TOTAL_GROUP):
        a=0
        b=0
        for s in range(0,TOTAL_SKILL):
            for i in range(0,group_need[0][g]):
                a += needed_skill[g,s]*res[s][i]
                b +=  needed_skill[g,s]*les[s][i]
        max_point_deep[g] = a
        min_point_deep[g] = b
    
    max_point_wise = {}
    min_point_wise = {}
    for g in range (0,TOTAL_GROUP):
        a=0
        b=0
        for s in range(0,TOTAL_SKILL):
            for i in range(0,group_need[0][g]):
                a += min(1, needed_skill[g,s]*res[s][i])
                b += min(1, needed_skill[g,s]*les[s][i])
        max_point_wise[g] = a
        min_point_wise[g] = b
   
    distance_deep = 0
    distance_wise = 0
    print (max_point_wise)
    print(min_point_wise)
    for g in range(0,TOTAL_GROUP):
       distance_deep += Weight[g]**2*(((max_point_deep[g] - deep[g])/(max_point_deep[g]-min_point_deep[g]))**1/2)
       distance_wise += Weight[g]**2*(((max_point_wise[g] - wise[g])/(max_point_wise[g]-min_point_wise[g]))**1/2)
        
    
    
    mdl.add_constraint(mdl.sum(x[i,j] for j in range(0,TOTAL_GROUP) for i in range(0,TOTAL_CANDIDATE) ) == 6)
    mdl.add_constraints(mdl.sum(x[i,j] for j in range(0,TOTAL_GROUP)) <= 1  for i in range(0,TOTAL_CANDIDATE))
    mdl.add_constraints(mdl.sum(x[i,g]*candidate_skill[i][s] for i in range(0,TOTAL_CANDIDATE)) == requirement[g][s] for g in range(0,TOTAL_GROUP) for s in range(0,TOTAL_SKILL))
    mdl.minimize_static_lex([distance_deep,distance_wise])
    # mdl.minimize(distance_deep)
    
    
   
    
    
    return mdl
def checkContraint1(solution):
    solution = np.array(solution)
    solution = solution.astype(np.int32)
    for i in range(0,TOTAL_GROUP):
        sum = 0
        for j in range(0,TOTAL_CANDIDATE):
            sum+= solution[j][i]
        if(sum>1): return False
    return True
def checkContraint2 (solution):
    solution = np.array(solution)
    solution = solution.astype(np.int32)
    sum = 0
    for i in range(0,TOTAL_GROUP):
        
        for j in range(0,TOTAL_CANDIDATE):
            sum+= solution[j][i]
    if(sum!=6): return False    
    return True
def bruteforce():
    requirement = np.array(REQUIREMENT)
    requirement = requirement.astype(np.float)
    a = np.array(CANDIDATE_SKILL)
    SKILL_CANDIDATE = a.transpose()
    a1 = np.array(SKILL_CANDIDATE)
    skill_candidate = np.delete(a1,0,0)    
    skill_candidate = skill_candidate.astype(np.float)
    les = np.sort(skill_candidate, axis = 1)
    
    
    res =-np.sort(-skill_candidate, axis = 1)
    
    group_need = np.array(GROUP)
    group_need = np.delete(group_need,0,1)
    group_need = group_need.transpose()   
    group_need=group_need.astype(np.int32) 
    
    Weight = [0.2,0.4,0.2,0.4]
    #model
    mdl = Model("MCDM", **kwargs)
    candidate_skill = np.array(CANDIDATE_SKILL)
    
    a = np.array(NEEDED_SKILL)
    candidate_skill = np.array(CANDIDATE_SKILL)
    candidate_skill = np.delete(candidate_skill,0,1)
    candidate_skill =candidate_skill.astype(np.float)  
    
    needed_skill = np.array(NEEDED_SKILL)
    needed_skill = np.delete(needed_skill,0,1)
    needed_skill = needed_skill.astype(np.float)
    
    # decision variables
    x = mdl.binary_var_matrix(TOTAL_CANDIDATE,TOTAL_GROUP,lambda ij: "x_%d_%d" %(ij[0], ij[1]))
    #deep objective
    deep = {}
    for j in range(0,TOTAL_GROUP):
        deep_obj = 0
        for n in range(0,TOTAL_SKILL):
            for i in range(0,TOTAL_CANDIDATE):         
               deep_obj =  mdl.sum(x[i,j]*candidate_skill[i][n]*needed_skill[j][n])
        deep[j]=deep_obj
    #wide objective
    wise = {}
    for j in range(0,TOTAL_GROUP):
        wise_obj = 0
        for n in range(0,TOTAL_SKILL):
            for i in range(0,TOTAL_CANDIDATE): 
               
               wise_obj =  mdl.sum(x[i,j]*min(1,candidate_skill[i][n]*needed_skill[j][n]))
        wise[j]=wise_obj
    max_point_deep = {}
    min_point_deep = {}
    
    for g in range (0,TOTAL_GROUP):
        a=0
        b=0
        for s in range(0,TOTAL_SKILL):
            for i in range(0,group_need[0][g]):
                a += needed_skill[g,s]*res[s][i]
                b +=  needed_skill[g,s]*les[s][i]
        max_point_deep[g] = a
        min_point_deep[g] = b
    
    max_point_wise = {}
    min_point_wise = {}
    for g in range (0,TOTAL_GROUP):
        a=0
        b=0
        for s in range(0,TOTAL_SKILL):
            for i in range(0,group_need[0][g]):
                a += min(1, needed_skill[g,s]*res[s][i])
                b += min(1, needed_skill[g,s]*les[s][i])
        max_point_wise[g] = a
        min_point_wise[g] = b
   
    distance_deep = 0
    distance_wise = 0
    print (max_point_wise)
    print(min_point_wise)
    for g in range(0,TOTAL_GROUP):
       distance_deep =mdl.sum(Weight[g]**2*(((max_point_deep[g] - deep[g])/(max_point_deep[g]-min_point_deep[g]))**1/2))
    #    distance_wise += Weight[g]**2*(((max_point_wise[g] - wise[g])/(max_point_wise[g]-min_point_wise[g]+1))**1/2)
        
    
    
    mdl.add_constraint(mdl.sum(x[i,j] for j in range(0,TOTAL_GROUP) for i in range(0,TOTAL_CANDIDATE) ) == 6)
    mdl.add_constraints(mdl.sum(x[i,j] for J in range(0,TOTAL_GROUP)) <= 1  for j in range(0,TOTAL_CANDIDATE))
    mdl.add_constraints(mdl.sum(x[i,g]*candidate_skill[i][s] for i in range(0,TOTAL_CANDIDATE)) == requirement[g][s] for g in range(0,TOTAL_GROUP) for s in range(0,TOTAL_SKILL))
    mdl.minimize_static_lex([distance_deep,distance_wise])
    # mdl.minimize(distance_deep)
    
    
   
    
    
    return mdl


if __name__ == '__main__':
    model = build_model()
    s=model.solve(log_output=True)
    # # Solve the model and print solution
    # model.solve()

    # # Save the CPLEX solution as "solution.json" program output
    if s:
        model.report()
        model.print_solution(log_output = True)
       
       
        # Save the CPLEX solution as "solution.json" program output
        with get_environment().get_output_stream("solution.json") as fp:
            model.solution.export(fp, "json")
    else:
        print("Problem has no solution")
    # a = np.zeros((TOTAL_CANDIDATE,TOTAL_GROUP))
    # print(checkContraint2(a))
    
    
   


    


        