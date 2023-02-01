# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
from data.dag_generator import DagGenerator

random.seed(10)
np.random.seed(10)

dag= DagGenerator('linear', cause='gaussian', nodes=10, expected_density= 0.5, npoints= 5000)
print(dir(dag))

df1, obj1= dag.generate()
z= df1.values
print('Observational Data')
print(z)
# print(np.cov(np.transpose(z)))
# print(z.shape)
# print(df1)
# print(df1.values.shape)
nx.draw_networkx(obj1, arrows=True)
plt.savefig('a.jpg')
plt.clf()
# sys.exit()

print('Adjanceny Matrix')
print(nx.adjacency_matrix(obj1))

print('Degree Values')
print(type(obj1.degree()))

# '''
# Calling dag.generate second time does not reinitialise the DAG structure but samples new points; so simply call dag.generate() with different seed values should give the train/val/test split.
# '''
# df2, obj2= dag.generate()
# print(df2)
# # nx.draw_networkx(obj2, arrows=True)
# # plt.savefig('b.jpg')
# # plt.clf()


#Checking for intervention
df3, obj3= dag.intervene(intervention_nodes= [9], target_distribution= 'hard_intervention')
print('Interventional Matrix')
print(df3)
# nx.draw_networkx(obj3, arrows=True)
# plt.savefig('c.jpg')
# plt.clf()