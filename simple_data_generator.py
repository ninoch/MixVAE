import sys
import random
import pickle
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# G: 78000.0 
# A1: 18000.0 
# A2: 60000.0

def write_edge_list(A, file_name):
	G = nx.from_numpy_matrix(A)
	d = {n: G.neighbors(n) for n in G.nodes()}
	pickle.dump(d, open(file_name, "wb"))

if __name__ == '__main__':
	print (sys.argv)
	file_name = sys.argv[1] 

	n = 40
	m = 4
	num_graphs = 50 

	cols = np.zeros(n, dtype=np.int32)
	for ind in range(n):
		cols[ind] = int((ind / n) * m)

	adj = np.zeros((n * num_graphs, n * num_graphs))
	y1 = np.zeros((n * num_graphs, n * num_graphs))
	y2 = np.zeros((n * num_graphs, n * num_graphs))
	feat = np.zeros((n * num_graphs, m))
	for ind in range(num_graphs):

		clus1 = np.copy(cols)
		random.shuffle(clus1)

		G_mat = np.ones((n, n)) - np.eye(n)
		A1_mat = np.zeros((n, n))
		for u in range(n):
			for v in range(n):
				if u != v and clus1[u] == clus1[v]:
					A1_mat[u][v] = 1
		A2_mat = G_mat - A1_mat 

		features = np.zeros((n, m), dtype=np.int32)
		features[range(n), list(clus1)] = 1
		features = features + np.random.normal(0.0, 0.01, n * m).reshape(n, m)

		adj[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n] = G_mat 
		y1[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n]  = A1_mat 
		y2[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n]  = A2_mat 
		feat[ind*n:(ind + 1)*n, :] = features

		print ("\tNumber of nodes = {}".format(G_mat.shape))
		print ("\tNumber of edges = {} (A1 = {}, A2 = {})".format(sum(G_mat), sum(A1_mat), sum(A2_mat)))
		print ("---------------------------------------------")

		# plt.clf()

		# G = nx.from_numpy_matrix(G_mat)
		# edge_col = [A1_mat[u][v] for u, v in G.edges()]

		# print (edge_col)

		# nx.draw(G, pos=nx.spring_layout(G), node_color=clus1, edge_color=edge_col, node_size=40)
		# plt.show() 

print (adj.shape)
print (y1.shape)
print (y2.shape)
print (np.sum(adj), np.sum(y1), np.sum(y2))

write_edge_list(adj, "data/synthetic/{}.graph".format(file_name))
write_edge_list(y1,  "data/synthetic/{}.y1".format(file_name))
write_edge_list(y2,  "data/synthetic/{}.y2".format(file_name))
pickle.dump(feat, open("data/synthetic/{}.x".format(file_name), "wb"))

