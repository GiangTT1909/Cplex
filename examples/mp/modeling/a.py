from math import sqrt

from docplex.mp.model import Model
from docplex.util.environment import get_environment

N = 15
box_range = range(1, N+1)
obj_range = range(1, N+1)

import random

o_xmax = N*10
o_ymax = 2*N
box_coords = {b: (10*b, 1) for b in box_range}

obj_coords= {1: (140, 6), 2: (146, 8), 3: (132, 14), 4: (53, 28), 
             5: (146, 4), 6: (137, 13), 7: (95, 12), 8: (68, 9), 9: (102, 18), 
             10: (116, 8), 11: (19, 29), 12: (89, 15), 13: (141, 4), 14: (29, 4), 15: (4, 28)}

# the distance matrix from box i to object j
# actually we compute the square of distance to keep integer
# this does not change the essence of the problem
distances = {}
for o in obj_range:
    for b in box_range:
        dx = obj_coords[o][0]-box_coords[b][0]
        dy = obj_coords[o][1]-box_coords[b][1]
        d2 = dx*dx + dy*dy
        distances[b, o] = d2

from docplex.mp.environment import Environment
env = Environment()
env.print_information()

from docplex.mp.model import Model

mdl = Model("boxes")
x = mdl.binary_var_matrix(box_range, obj_range, lambda ij: "x_%d_%d" %(ij[0], ij[1]))
mdl.add_constraints(mdl.sum(x[i,j] for j in obj_range) == 1
                   for i in box_range)
    
# one box for each object
mdl.add_constraints(mdl.sum(x[i,j] for i in box_range) == 1
                  for j in obj_range)
print(mdl.sum(distances[i,j] * x[i,j] for i in box_range for j in obj_range))

mdl.minimize( mdl.sum(distances[i,j] * x[i,j] for i in box_range for j in obj_range) )
print(mdl.sum(distances[i,j] * x[i,j] for i in box_range for j in obj_range))
mdl.solve()
