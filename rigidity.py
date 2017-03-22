import numpy
import numpy.linalg
import scipy.special


def rigidity_matrix(vertex_config, edges):
    """
    Compute the rigidity matrix as a numpy array.
    Each row corresponds to the constraint imposed by a single edge,
    in the order given in edges.
    If C is the mapping from vertices to coordinates,
    the convention is that, for each row (corresponding to edge (u,v),
    the coordinates in the columns for u are C(u) - C(v);
    for v, C(v) - C(u).

    vertex_config: a 2D array-like of shape n, d, where n is the number of
                   vertices and d is the dimension of the embedding space
    edges: an iterable of linkage edges, specified as integer pairs.
    """
    vertex_vectors = numpy.array(vertex_config)
    vertex_n, embed_dimension = vertex_vectors.shape

    rigidity_matrix_rows = []
    for (left, right) in edges:
        edge_displacement = vertex_vectors[right] - vertex_vectors[left]

        rigidity_mat_row = numpy.zeros(vertex_n * embed_dimension)
        left_ind = embed_dimension * left
        right_ind = embed_dimension * right
        rigidity_mat_row[left_ind:left_ind+embed_dimension] = -edge_displacement
        rigidity_mat_row[right_ind:right_ind+embed_dimension] = edge_displacement

        rigidity_matrix_rows.append(rigidity_mat_row)

    return numpy.array(rigidity_matrix_rows)

def inf_dof(vertex_config, edges):
    """
    Compute the number of infinitesimal degrees of freedom
    of a linkage in a specified configuration.

    vertex_config: A list of vertex coordinates, specified as equal-length tuples.
    edges: an iterable of linkage edges,
           specified as integer pairs (vertex1, vertex2)
           that index into vertex_config
    """
    if len(vertex_config) == 0:
        return 0

    vertex_vectors = numpy.array(vertex_config)
    vertex_n, embed_dimension = vertex_vectors.shape

    rmat_rank = numpy.linalg.matrix_rank(rigidity_matrix(vertex_vectors, edges))

    # dimension k of lowest-dimensional hyperplane containing all vertices
    displacements = vertex_vectors - vertex_vectors[0]
    configuration_dim = numpy.linalg.matrix_rank(displacements)

    # n*d - rank = nullity = (dim infinitesimal rigid motions) + dof
    # dim infinitesimal rigid motions = dim SO(d) - dim symmetry group of vertices
    # dim symmetry group of vertices = dim SO(d - k)
    # dim SO(m) = m+1 choose 2

    euclidean_isometries_dim = int(scipy.special.binom(embed_dimension + 1, 2))
    symmetry_group_dim = int(scipy.special.binom(embed_dimension - configuration_dim, 2))

    return embed_dimension*vertex_n \
            - rmat_rank \
            - euclidean_isometries_dim \
            + symmetry_group_dim


def is_generically_rigid(n_vertices, edges, dimensions):
    """
    A randomized algorithm to test generic ridigity of a graph in any dimension.

    n_vertices: number of vertices
    edges: specified as in inf_dof
    dimensions: embedding dimension
    """
    amplification_iterations = 1  # increase for higher probability of success (which is pretty high anyway)

    for iteration in range(amplification_iterations):
        random_vertex_config = numpy.random.rand(n_vertices, dimensions)
        if inf_dof(random_vertex_config, edges) == 0:
            return True
    return False
