import numpy
import numpy.linalg

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

	# we build up the rigidity matrix row by row
	rigidity_mat_rows = []
	for (left, right) in edges:
		edge_displacement = \
			numpy.array(vertex_config[left]) - numpy.array(vertex_config[right])

		rigidity_mat_row = numpy.zeros(vertex_n * embed_dimension)
		left_ind = embed_dimension * left
		right_ind = embed_dimension * right
		rigidity_mat_row[left_ind:left_ind+embed_dimension] = edge_displacement
		rigidity_mat_row[right_ind:right_ind+embed_dimension] = edge_displacement
		
		rigidity_mat_rows.append(rigidity_mat_row)

	# n*d - rank = nullity = \binom{d+1}{2} + dof
	rmat_rank = numpy.linalg.matrix_rank(rigidity_mat_rows)
	plane_isometries_dim = (embed_dimension + 1) * embed_dimension // 2
	return embed_dimension*vertex_n - rmat_rank - plane_isometries_dim
