#! /usr/bin/env python

import sys
import os
import optparse
import re

def median(ls):
    d = {}
    for x in ls:
        if x in d: d[x] += 1
        else:      d[x] = 1
    m = max(d.keys(), key=lambda x: d[x])
    return (m, d[m])

class Evaluator:
    """
    
    """
    def __init__(self, reference, encoding='UTF-8', verbose=False):
        """
        @param  reference:  file or dict that contains reference classifications
        @type   reference:  file
        """
        
        self.verbose=verbose
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
        for (i,c) in self.reference.items():
            for x in c:
                self.cmemberships.append((i, x))
    def evaluated_docs(self, memberships):
        return filter(lambda x: x[0] in self.reference, dict(memberships).keys())
    def purity(self, memberships, macro=True, verbose=None):
        verbose = self.verbose if verbose is None else False
        return Evaluator.__purity(memberships, self.reference, macro=macro, verbose=verbose)

    def inverse_purity(self, memberships, macro=True, verbose=None):
        verbose = self.verbose if verbose is None else False
        d = dict(memberships)
        d2 = dict(self.cmemberships)
        return Evaluator.__purity(filter(lambda x: x[0] in d, self.cmemberships),
                                  dict([(x[0],[x[1]]) for x in memberships if x[0] in d2]),
                                  macro=macro, verbose=verbose)

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
            (m,count) = median(reduce(lambda s,x: s+x, [references[x] for x in docs]))
            val = float(count)/float(len(docs))
            if verbose:
                print '#', val, count, len(docs), docs, reduce(lambda s,x: s+x, [references[x] for x in docs])
            if macro:
                avg += val*len(docs)
            else:
                avg += val
        if macro:
            return avg / len(memberships)
        else:
            return avg / len(clusters)
    
    @classmethod
    def read_membership_file(cls,file,encoding='UTF-8'):
        m = []
        for line in file:
            p = line.rfind('#')
            if p == -1:  p = len(line)
            v = re.split(r'\s+', line[0:p].strip())
            if len(v) >= 2:
                (id,cats) = (v[0], v[1:])
                cats = [x.decode(encoding) for x in cats]
                m.append((id,cats[0]))
        return m


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-m', '--method', metavar='METHOD',
                      dest='method', type=str, default='svd',
                      help='number of words to extract')
    parser.add_option('-e', '--encoding', metavar='ENCODING',
                      dest='encoding', default='utf-8',
                      help='input/output encoding')
    parser.add_option('-v', '--verbose', metavar='VERBOSE',
                      dest='verbose', action='store_true', default=False,
                      help='turn on verbose message output')
    (options, args) = parser.parse_args()
    (catfile,resfiles) = tuple([args[0],args[1:]])
    if options.verbose:
        print options, catfile, resfiles
    eval = Evaluator(open(catfile), options.encoding, verbose=options.verbose)
    print '\t%s\t%s\t%s\t%s\t%s' % ('num.', 'ma. pur.', 'ma. i. pur.', 'mi. pur.', 'mi. i. pur.')
    for resfile in resfiles:
        memberships = Evaluator.read_membership_file(open(resfile),options.encoding)
        print '%s\t%d/%d\t%f\t%f\t%f\t%f' % (
            resfile,
            len(eval.evaluated_docs(memberships)), len(memberships),
            eval.purity(memberships),
            eval.inverse_purity(memberships),
            eval.purity(memberships,macro=False),
            eval.inverse_purity(memberships,macro=False))
