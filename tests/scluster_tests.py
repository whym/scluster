#! /usr/bin/env python

import unittest
from tempfile import mktemp, mkdtemp
from six import StringIO
from scluster import clusterer, evaluate
import numpy as np

class TestDocCluster(unittest.TestCase):

    def setUp(self):
        pass

    def test_doc2vectors(self):
        pass

    def test_kmeans(self):
        mem, cen = clusterer.kmeans(
            np.array([[0.01, 0.5],
                      [0.02, 0.5],
                      [0.9, 0.1],
                      [0.85, 0.2],
                      [0.8, 0.1]]),
            np.array([[0.0, 0.0],
                      [1.0, 1.0]])
            )
        self.assertTrue(np.all(np.array([0, 0, 1, 1, 1]) == mem))


class TestEvaluate(unittest.TestCase):
    
    def setUp(self):
        pass

    def testMedian(self):
        self.assertEqual((4,3), evaluate.median([4,1,2,4,4,3,3]))

    def testUniq(self):
        self.assertEqual([4,1,2,3], evaluate.uniq([4,1,2,4,4,3,3]))

    def testPurity(self):
        self.__testPurity({'1': ['11'], '2': ['13'], '3': ['11'], '4': ['13'], '5': ['13']})

        self.__testPurity(StringIO("""
1 11
2 13
3 11
4 13
5 13
"""
                              ))
    
    def __testPurity(self, input):
        ev = evaluate.Evaluator(input)
        self.assertEqual(1.0, ev.purity({'1': '11', '2': '13', '3': '11', '4': '13', '5': '13'}.items()))
        self.assertEqual(0.6, ev.purity({'1': '13', '2': '11', '3': '11', '4': '13', '5': '13'}.items(), macro=True))


if __name__ == '__main__':
    unittest.main()
