import unittest
import math
import numpy
import numpy.testing
from inf_rigidity_testing import rigidity_matrix, inf_dof

class RigidityMatrixTest(unittest.TestCase):
    def testPage50Example(self):
        """
        The example from page 50 of the GFALOP e-book.
        """
        config = [(0,0),(5,0),(5,5),(0,5)]
        edges = [(0,1), (1,2), (2,3), (3,0)]
        expected_rigiditiy_matrix = numpy.array([
            [-5,  0, 5,  0, 0, 0,  0, 0],
            [ 0,  0, 0, -5, 0, 5,  0, 0],
            [ 0,  0, 0,  0, 5, 0, -5, 0],
            [ 0, -5, 0,  0, 0, 0,  0, 5]
        ])
        # numpy array equivalent of assertEqual
        numpy.testing.assert_allclose(rigidity_matrix(config, edges),
                                      expected_rigiditiy_matrix)

class InfinitesimalDofTest(unittest.TestCase):
    def testLine(self):
        config = [(0,0),(1,0)]
        edges = [(0,1)]
        self.assertEqual(inf_dof(config, edges), 0)

    def testTriangle(self):
        triangle_config = [(0,0), (1,0), (0,1)]
        edges = [(0,1), (1,2), (2,0)]
        self.assertEqual(inf_dof(triangle_config, edges), 0)

    def testSquare(self):
        square_config = [(0,0),(0,5),(5,5),(5,0)]
        edges = [(0,1), (1,2), (2,3), (3,0)]
        self.assertEqual(inf_dof(square_config, edges), 1)

    def testTwoSquares(self):
        """
        1--3--5
        |  |  |
        0--2--4
        """
        two_square_config = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
        edges = [(0,1), (2,3), (4,5), (0,2), (2,4), (1,3), (3,5)]
        self.assertEqual(inf_dof(two_square_config, edges), 2)

    def testSpokes(self):
        n_spokes = 50
        angle = 2 * math.pi / n_spokes
        spoke_config = [(0,0)] + [(math.cos(k * angle), math.sin(k * angle))
                            for k in range(n_spokes)]
        edges = [(0, k + 1) for k in range(n_spokes)]
        # minus 1 because uniform rotation is a plane isometry
        self.assertEqual(inf_dof(spoke_config, edges), n_spokes - 1)

    def test3DTriangle(self):
        triangle_config = [(0,0,1), (1,0,1), (0,1,1)]
        edges = [(0,1), (1,2), (2,0)]
        self.assertEqual(inf_dof(triangle_config, edges), 0)

    def test3DSquare(self):
        square_config = [(0,0,100),(0,5,100),(5,5,100),(5,0,100)]
        edges = [(1,2),(2,3),(3,0),(0,1)]

        # extra dof because we can fold now.
        self.assertEqual(inf_dof(square_config, edges), 2)

    def testCube(self):
        """
        2--3              6--7
        |  | connected to |  |
        0--1              4--5

        There are 6 dof.
        """
        cube_config = [(0,0,0),(1,0,0),
                       (0,1,0),(1,1,0),
                       (0,0,1),(1,0,1),
                       (0,1,1),(1,1,1)]
        edges = [(0,1),(1,3),(3,2),(2,0),
                 (0,4),(1,5),(2,6),(3,7),
                 (4,5),(5,7),(7,6),(6,4)]
        self.assertEqual(inf_dof(cube_config, edges), 6)

    def testNDPoint(self):
        dim = 79
        config = [(0,)*dim]
        edges = []
        self.assertEqual(inf_dof(config, edges), 0)

    def testNDLine(self):
        dim = 17
        config = [(0,)*dim, (1,)*dim]
        edges = [(0,1)]
        self.assertEqual(inf_dof(config, edges), 0)

if __name__ == '__main__':
    unittest.main()