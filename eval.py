#! /usr/bin/env python

import sys
import os
import optparse
import re

class Evaluator:
    """
    
    """
    def __init__(self, reference, encoding='UTF-8'):
        """
        @param reference:  file or dict that contains reference classifications
        """
        
        if type(reference) is file:
            self.reference = {}
            for line in reference:
                v = re.split(r'\s+', line.strip())
                if len(v) >= 2:
                    (id,cats) = (v[0], v[1:])
                    cats = [x.decode(encoding) for x in cats]
                    self.reference[id] = cats
        elif type(reference) is dict:
            self.reference = reference
        self.cmemberships = []
        for (i,c) in enumerate(self.reference):
            for x in c:
                self.cmemberships.append((i, x))

    def purity(self, memberships, macro=True, verbose=False):
        return Evaluator.__purity(memberships, self.reference, macro=macro, verbose=verbose)

    def inverse_purity(self, memberships, macro=True, verbose=False):
        return Evaluator.__purity(self.cmemberships.items(), [(x[0],[x[1]]) for x in memberships], macro=macro, verbose=verbose)

    @classmethod
    def __purity(cls, memberships, references, macro=True, verbose=False):
        clusters = {}
        avg = 0.0
        for (i,x) in memberships:
            if x in clusters:
                clusters[x].append(i)
            else:
                clusters[x] = [i]
        for docs in clusters.values():
            m = sorted(reduce(lambda s,x: s+x, [references[x] for x in docs]))
            count = len(filter(lambda x: x==m[0], m))
            val = float(count)/float(len(docs))
            if verbose:
                print '#', val, count, len(docs)
            if macro:
                avg += val*len(docs)
            else:
                avg += val
        if macro:
            return avg / len(memberships)
        else:
            return avg / len(clusters)
    
    @classmethod
    def read_membership_file(file):
        m = []
        for line in file:
            pass
