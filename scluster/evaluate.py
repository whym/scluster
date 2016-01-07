#! /usr/bin/env python

import argparse
import re
import math
import codecs
from six.moves import reduce


def uniq(ls):
    d = {}
    ret = []
    for x in ls:
        if x not in d:
            d[x] = True
            ret.append(x)
    return ret


# I know numpy has these functions, but I wanted to make this script numpy-free
def median(ls):
    d = {}
    for x in ls:
        d[x] = d.get(x, 0) + 1
    m = max(d.keys(), key=lambda x: d[x])
    return (m, d[m])


def avg(ls, map=lambda x: x):
    return float(reduce(lambda s, x: s + map(x), ls)) / float(len(ls))


def deviation(ls, order=2):
    a = avg(ls)
    return avg(ls, map=lambda x: (x - a) ** order)


class Evaluator:
    """
    """
    def __init__(self, reference, verbose=False):
        """
        Construct a new evaluator.  The first argument can be either a
        dict or a file that contains the mapping from document id to categories.
        The format of the files:
        <document-id> <cluster-id> <cluster-id> ...
        <document-id> <cluster-id> <cluster-id> ...
        ..

        @param  reference:  file or dict that contains reference classifications
        @type   reference:  file
        """

        self.verbose = verbose
        if type(reference) is dict:
            self.reference = reference
        else:
            self.reference = {}
            for line in reference:
                v = re.split(r'\s+', line.strip())
                if len(v) >= 2:
                    (id, cats) = (v[0], v[1:])
                    self.reference[id] = uniq(cats)
        self.cmemberships = []
        for (i, c) in self.reference.items():
            for x in c:
                self.cmemberships.append((i, x))

    def evaluated_docs(self, memberships):
        return Evaluator.__intersect(memberships, self.reference)

    def purity(self, memberships, macro=True, verbose=None):
        verbose = self.verbose if verbose is None else False
        (m, r) = Evaluator.__intersect(memberships, self.reference)
        return Evaluator.__purity(m, r, macro=macro, verbose=verbose)

    def inverse_purity(self, memberships, macro=True, verbose=None):
        verbose = self.verbose if verbose is None else False
        (m, r) = Evaluator.__intersect(
            self.cmemberships,
            dict([(x[0], [x[1]]) for x in memberships])
        )
        return Evaluator.__purity(m, r, macro=macro, verbose=verbose)

    def sizes(self, memberships):
        sizes = {}
        for (x, c) in memberships:
            if c in sizes:
                sizes[c] += 1
            else:
                sizes[c] = 1
        return sizes.values()

    @classmethod
    def __intersect(cls, memberships, reference):
        mem = dict(memberships)
        ret2 = {}
        for x in reference.keys():
            if x in mem:
                ret2[x] = reference[x]
        return ([x for x in mem.items() if str(x[0]) in reference],
                ret2)

    @classmethod
    def __purity(cls, mem, references, macro=True, verbose=False):
        clusters = {}
        avg = 0
        memberships = [x for x in mem]
        for (i, x) in memberships:
            if x in clusters:
                clusters[x].append(i)
            else:
                clusters[x] = [i]
        for docs in clusters.values():
            (_, count) = median(reduce(lambda s, x: s + x, [references[x] for x in docs]))
            if verbose:
                print('# %f = %d/%d, labels %s for cluster %s' % (
                    float(count) / float(len(docs)),
                    count, len(docs),
                    str(reduce(lambda s, x: s + x, [references[x] for x in docs])),
                    str(docs)))
            if macro:
                avg += count
            else:
                avg += float(count) / float(len(docs))
        if macro:
            return float(avg) / len(memberships)
        else:
            return float(avg) / len(clusters)

    @classmethod
    def read_membership_file(cls, ofile):
        m = []
        for line in ofile:
            p = line.rfind('#')
            if p < 0:
                p = len(line)
            v = re.split(r'\s+', line[0:p].strip())
            if len(v) >= 2:
                (id, cats) = (v[0], v[1:])
                m.append((id, cats[0]))
        return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoding', metavar='ENCODING',
                        dest='encoding', default='utf-8',
                        help='input/output encoding')
    parser.add_argument('-v', '--verbose',
                        dest='verbose', action='store_true', default=False,
                        help='turn on verbose message output')
    parser.add_argument('catfile')
    parser.add_argument('resfiles', nargs='+')
    args = parser.parse_args()
    if args.verbose:
        print('%s' % (args))
    eval = Evaluator(
        codecs.open(args.catfile, encoding=args.encoding),
        verbose=args.verbose
    )
    print('\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
        'num.', 'num. cl', 'avg. size', 'dev. size',
        'ma. pur.', 'ma. i. pur.', 'mi. pur.', 'mi. i. pur.')
    )
    for resfile in args.resfiles:
        memberships = Evaluator.read_membership_file(
            codecs.open(resfile, encoding=args.encoding)
        )
        print('%s\t%d/%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f' % (
            resfile,
            len(eval.evaluated_docs(memberships)[0]), len(memberships),
            len(eval.sizes(memberships)),
            avg(eval.sizes(memberships)),
            math.sqrt(deviation(eval.sizes(memberships))),
            eval.purity(memberships),
            eval.inverse_purity(memberships),
            eval.purity(memberships, macro=False),
            eval.inverse_purity(memberships, macro=False)))
