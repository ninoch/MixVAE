import sys
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


def get_stochastic_block_matrix_edges(n, clus, p, q):
	edges = []
	probs = np.random.uniform(low=0.0, high=1.0, size=int((n * (n - 1)) / 2))
	cnt = 0
	for u in range(n):
		for v in range(n):
			if (u < v):
				e_prob = probs[cnt]
				cnt += 1
				if clus[u] == clus[v] and e_prob < p:
						edges.append((u, v))
				elif clus[u] != clus[v] and e_prob < q:
						edges.append((u, v))

	# Drawing histogram of probabilities 
	# plt.hist(probs, bins=20)
	# plt.axvline(x = p, color = 'red')
	# plt.axvline(x = q, color = 'red')
	# plt.show()

	return edges



if __name__ == '__main__':
	"""
	Example Run: 

	Where: 
		n = number of nodes
		m = number of components
		p1 = probability of inner-cluster edge for features graph
		q1 = probability of intra-cluster edge for features graph
		p2 = probability of inner-cluster edge for structure graph
		q2 = probability of intra-cluster edge for structure graph
		num_graphs = number of graph samples
		file_name = prefix to save samples to file
	"""
	print (sys.argv)
	n, m, p1, q1, p2, q2, num_graphs, file_name = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]

	print ("Generating block-matrix graph with \n\t{} nodes, \n\t{} components, \n\tfeature inner-cluster probability = {}, \n\tfeature intra-cluster probability = {}\n\tstructure inner-cluster probability = {}, \n\tstructe intra-cluster probability = {}".format(n, m, p1, q1, p2, q2))
	print ("---------------------------------------------")
	clus2 = np.zeros(n, dtype=np.int32)
	for ind in range(n):
		clus2[ind] = int((ind / n) * m)

	adj = np.zeros((n * num_graphs, n * num_graphs))
	y1 = np.zeros((n * num_graphs, n * num_graphs))
	y2 = np.zeros((n * num_graphs, n * num_graphs))
	feat = np.zeros((n * num_graphs, m))
	for ind in range(num_graphs):

		clus1 = np.copy(clus2)
		random.shuffle(clus1)


		A1_edges = get_stochastic_block_matrix_edges(n, clus1, p1, q1)
		A2_edges = get_stochastic_block_matrix_edges(n, clus2, p2, q2)

		A1 = nx.Graph()
		A1.add_nodes_from(range(n))
		A1.add_edges_from(A1_edges, color='lightgray')

		A1_mat = nx.adjacency_matrix(A1).todense()
		# print (A1_mat)

		A2 = nx.Graph()
		A2.add_nodes_from(range(n))
		A2.add_edges_from(A2_edges, color='lightgray')

		A2_mat = nx.adjacency_matrix(A2).todense()
		# print (A2_mat)

		# print (A1_mat + A2_mat)


		# print ("\tFeatures: ")
		features = np.zeros((n, m), dtype=np.int32)
		features[range(n), list(clus1)] = 1

		# print (features)
		features = features + np.random.normal(0.0, 0.01, n * m).reshape(n, m)
		# print (features)

		G = nx.Graph()
		G.add_nodes_from(range(n))
		G.add_edges_from(A1_edges, color='lightgray')
		G.add_edges_from(A2_edges, color='black')

		adj[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n] = nx.adjacency_matrix(G).todense() 
		y1[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n]  = nx.adjacency_matrix(A1).todense() 
		y2[ind*n:(ind + 1)*n, ind*n:(ind + 1)*n]  = nx.adjacency_matrix(A2).todense() 
		feat[ind*n:(ind + 1)*n, :] = features


		# Drawing Graph
		# plt.clf()

		# edge_col = [G[u][v]['color'] for u, v in G.edges()]

		# nx.draw(G, pos=nx.spring_layout(G), node_color=clus2, edge_color=edge_col, node_size=40)
		# plt.show() 

		print ("\tNumber of nodes = {}".format(len(G.nodes())))
		print ("\tNumber of edges = {} (A1 = {}, A2 = {})".format(len(G.edges()), len(A1_edges), len(A2_edges)))
		print ("---------------------------------------------")

print (adj.shape)
print (y1.shape)
print (y2.shape)
print (np.sum(adj), np.sum(y1), np.sum(y2))

pickle.dump(adj, open("data/synthetic/{}.graph".format(file_name), "wb"))
pickle.dump(feat, open("data/synthetic/{}.allx".format(file_name), "wb"))
pickle.dump(y1, open("data/synthetic/{}.ally1".format(file_name), "wb"))
pickle.dump(y2, open("data/synthetic/{}.ally2".format(file_name), "wb"))


