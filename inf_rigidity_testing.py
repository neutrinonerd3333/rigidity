import numpy
import numpy.linalg
import scipy.special

def displacement(start, finish):
	"""
	The vector difference finish - start.
	"""
	return numpy.array(finish) - numpy.array(start)

def inf_dof(vertex_config, edges):
	"""
	Compute the number of infinitesimal degrees of freedom
	of a linkage in a specified configuration.

	vertex_config: A list of vertex coordinates, specified as equal-length tuples.
	edges: an iterable of linkage edges,
		   specified as integer pairs (vertex1, vertex2)
		   that index into vertex_config
	"""

	embed_dimension = len(vertex_config[0])
	vertex_n = len(vertex_config)
	vertex_vectors = numpy.array(vertex_config)

	# we build up the rigidity matrix row by row
	rigidity_mat_rows = []
	for (left, right) in edges:
		edge_displacement = vertex_vectors[right] - vertex_vectors[left]

		rigidity_mat_row = numpy.zeros(vertex_n * embed_dimension)
		left_ind = embed_dimension * left
		right_ind = embed_dimension * right
		rigidity_mat_row[left_ind:left_ind+embed_dimension] = edge_displacement
		rigidity_mat_row[right_ind:right_ind+embed_dimension] = -edge_displacement
		
		rigidity_mat_rows.append(rigidity_mat_row)
	rmat_rank = numpy.linalg.matrix_rank(rigidity_mat_rows)

	# dimension k of lowest-dimensional hyperplane containing all vertices
	displacements = vertex_vectors - vertex_vectors[0]
	configuration_dim = numpy.linalg.matrix_rank(displacements)

	# n*d - rank = nullity = (dim infinitesimal rigid motions) + dof
	# dim infinitesimal rigid motions = dim SO(d) - dim symmetry group of vertices
	# dim symmetry group of vertices = dim SO(d - k)
	# dim SO(m) = m+1 choose 2

	euclidean_isometries_dim = scipy.special.binom(embed_dimension + 1, 2)
	symmetry_group_dim = scipy.special.binom(embed_dimension - configuration_dim, 2)

	return embed_dimension*vertex_n \
			- rmat_rank \
			- euclidean_isometries_dim \
            + symmetry_group_dim
