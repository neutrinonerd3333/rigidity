import numpy
import numpy.linalg

def inf_dof(vertex_config, edges):
	"""
	Compute the number of infinitesimal degrees of freedom
	of a linkage in a specified configuration.

	vertex_config: a dict {vertex: coordinates, ...}.
	coordinates must be tuples of equal length
	edges: an iterable of edges specified as vertex pairs (vertex1, vertex2)
	"""
	embed_dimension = len(next(iter(vertex_config.values())))
	vertex_n = len(vertex_config)
	vertices = list(vertex_config.keys())

	rigidity_mat_rows = []
	for (left, right) in edges:
		edge_displacement = \
			numpy.array(vertex_config[left]) - numpy.array(vertex_config[right])
		print(edge_displacement)
		rigidity_mat_row = numpy.zeros(vertex_n * embed_dimension)
		print(rigidity_mat_row)

	    ## TODO make more efficient instead of calling ``index"
		left_ind = embed_dimension * vertices.index(left)
		right_ind = embed_dimension * vertices.index(right)
		rigidity_mat_row[left_ind:left_ind+embed_dimension] = edge_displacement
		rigidity_mat_row[right_ind:right_ind+embed_dimension] = edge_displacement
		rigidity_mat_rows.append(rigidity_mat_row)

	rigidity_mat = numpy.array(rigidity_mat_rows)
	rmat_rank = numpy.linalg.matrix_rank(rigidity_mat)
	return embed_dimension * vertex_n - rmat_rank - (embed_dimension + 1) * embed_dimension // 2

result = inf_dof({1: (0,0), 2: (5,0), 3: (5,5), 4: (0,5)}, [(1,2),(2,3),(3,4),(4,1)])
print(result)