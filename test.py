#! /usr/bin/env python

import unittest
from tempfile import mktemp, mkdtemp
from StringIO import StringIO
import evaluate

class TestDocCluster(unittest.TestCase):

    def setUp(self):
        pass
    def test_doc2vectors(self):
        pass

class TestEvaluate(unittest.TestCase):
    
    def setUp(self):
        pass

    def testMedian(self):
        self.assertEqual((4,3), evaluate.median([4,1,2,4,4,3,3]))

    def testUniq(self):
        self.assertEqual([4,1,2,3], evaluate.uniq([4,1,2,4,4,3,3]))

    def testPurity(self):
        self.__testPurity({'1': [u'11'], '2': [u'13'], '3': [u'11'], '4': [u'13'], '5': [u'13']})

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
        self.assertEqual(1.0, ev.purity({'1': u'11', '2': u'13', '3': u'11', '4': u'13', '5': u'13'}.items()))
        self.assertEqual(0.6, ev.purity({'1': u'13', '2': u'11', '3': u'11', '4': u'13', '5': u'13'}.items(), macro=True))


if __name__ == '__main__':
    unittest.main()
