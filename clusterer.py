#! /usr/bin/env python

import sys
import os
import optparse
import re
import scipy
import scipy.linalg
from scipy import mat, zeros, shape, diag, dot, random
from math import log
try:
    from functools import reduce
except ImportError:
    pass
import time
from eval import Evaluator

def ng_matrix_epsilon(itemvectors, epsilon, distance=lambda x,y: scipy.linalg.norm(x-y,2), binary=False):
    c = 0
    weighting = lambda x: x
    if binary:
        weighting = lambda x: 1
    matrix = zeros((itemvectors.shape[0], itemvectors.shape[0]), float)
    for (i,x) in enumerate(itemvectors):
        for (j,y) in enumerate(itemvectors[i:]):
            d = distance(x,y)
            if d < epsilon:
                c += 1
                w = weighting(d)
                matrix[i][i+j] = w
                matrix[i+j][i] = w
    print >> sys.stderr, "ng_matrix_epsilon: number of edges =", c
    return matrix

def kmeans(itemvectors, initial, distance=lambda x,y: scipy.linalg.norm(x-y,2), threshold=1.0E-6, iterations=1000, verbose=False):
    centroids = initial
    memberships = zeros(len(itemvectors), int)
    old_objective = 0.0
    for it in range(0,iterations):
        objective = 0.0
        # renew memberships
        for (i,v) in enumerate(itemvectors):
            mi = (float(sys.maxint),None)
            for (j,c) in enumerate(centroids):
                d = distance(v,c)
                if d < mi[0]:
                    mi = (d, j)
                    memberships[i] = j
            objective += mi[0]
        if verbose:
            print objective
        if abs(objective - old_objective) < threshold:
            break
        old_objective = objective
        # renew centroids
        oldc = centroids
        centroids = zeros(centroids.shape, float)
        members = zeros(len(centroids), int)
        for (i,m) in enumerate(memberships):
            centroids[m] += itemvectors[i]
            members[m] += 1
        for (i,x) in enumerate(centroids):
            if members[i] == 0:
                print >> sys.stderr, members, memberships
                print >> sys.stderr, " cluster", i, "is empty"
                centroids[i] = oldc[i]
            else:
                centroids[i] /= members[i]
    
    return memberships, centroids

def random_kmeans_init(itemvectors, num_clusters):
    memberships = [x * num_clusters // len(itemvectors) for x in xrange(0,len(itemvectors))]
    random.shuffle(memberships)
    centroids = zeros((num_clusters, itemvectors.shape[1]), float)
    members = zeros(len(centroids), int)
    for (i,m) in enumerate(memberships):
        centroids[m] += itemvectors[i]
        members[m] += 1
    for (i,x) in enumerate(centroids):
        centroids[i] /= members[i]
    return (memberships,centroids)

def docs2vectors_tfidf(dir, vocsize, verbose=False):
    words = {}
    
    tf = []
    df = {}
    docs = os.listdir(dir)
    for file in docs:
        freq = {}
        print 'reading', file
        for line in open(dir + '/' + file):
            for w in re.split(r'\s+', line.strip()):
                if w not in words:
                    words[w] = 1
                else:
                    words[w] += 1
                if w not in freq:
                    freq[w] = 1
                else:
                    freq[w] += 1
                if w not in df:
                    df[w] = 1
                else:
                    df[w] += 1
        tf.append((file, freq))
    id2word = sorted(words.keys(), key=lambda x: df[x], reverse=True)[0:vocsize]
    word2id = {}
    mat = zeros((len(tf), len(id2word)), float)
    for (i,c) in enumerate(id2word):
        word2id[c] = i
    for (i,row) in enumerate(mat):
        for w in [ x for x in tf[i][1].keys() if x in word2id ]:
            row[word2id[w]] = log(1.0 + tf[i][1][w]) / df[w]
    return (mat, docs)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-e', '--encoding', metavar='ENCODING',
                      dest='encoding', default='utf-8',
                      help='input/output encoding')
    parser.add_option('-n', '--num-words', metavar='NUM_WORDS',
                      dest='num_words', type=int, default=5000,
                      help='number of words to extract')
    parser.add_option('-m', '--method', metavar='METHOD',
                      dest='method', type=str, default='svd,kmeans',
                      help="""clustering algorithms:
                              comma-separated combination of
                              svd,kmeans,spectral
                      """)
    parser.add_option('-o', '--output', metavar='OUTPUT',
                      dest='output', type=str, default=None,
                      help='output filename')
    parser.add_option('-d', '--dimension', metavar='DIMENSION',
                      dest='dimension', type=int, default=2000,
                      help='dimension of document vector')
    parser.add_option('-p', '--precision', metavar='PRECISION',
                      dest='precision', type=float, default=1.0E-6,
                      help='threshold for k-means clustering convergence')
    parser.add_option('-E', '--epsilon', metavar='EPSILON',
                      dest='epsilon', type=float, default=10.0,
                      help='threshold for constructing neiborhood graph')
    parser.add_option('-r', '--random-seed', metavar='SEED',
                      dest='seed', type=int, default=int(time.time()),
                      help='random number seed')
    parser.add_option('-K', '--num-clusters', metavar='CLUSTERS',
                      dest='clusters', type=int, default=40,
                      help='number of clusters')
    parser.add_option('-S', '--num-spectral-clusters', metavar='SCLUSTERS',
                      dest='sclusters', type=int, default=20,
                      help='number of clusters')
    parser.add_option('--binary-neighbourhood', metavar='BINARY_NEIGHBOUR',
                      dest='bneighbour', action='store_true', default=False,
                      help='use binary value for neighborhood')
    parser.add_option('-v', '--verbose', metavar='VERBOSE',
                      dest='verbose', action='store_true', default=False,
                      help='turn on verbose message output')
    (options, args) = parser.parse_args()
    (catfile,dir) = tuple(args[0:2])
    options.method = options.method.split(',')
    options.method = dict(zip(options.method, options.method))
    if options.verbose:
        print options, catfile, dir
    
    random.seed(options.seed)

    # obtain document vectors
    (docvectors,docids) = docs2vectors_tfidf(dir, options.num_words)

    if options.verbose:
        print '#documents: ', len(docvectors)

    eval = Evaluator(open(catfile), options.encoding)

    if 'svd' in options.method:
        # dimensionality reduction by svd
        (U,s,Vh) = scipy.linalg.svd(docvectors,full_matrices=False)
        #print shape(U),shape(s),shape(Vh)
        #print [x for x in docvectors[1]]
        docvectors = dot(U[:,:options.dimension], diag(s[:options.dimension]))

        #print [x for x in docvectors[1]]
        if options.verbose:
            print [x for x in s]
    #TODO: compare to random indexing
    
    (memberships,centroids) = random_kmeans_init(docvectors, options.clusters)
    print  'initial centroids: \n', centroids

    print '     purity: ', eval.purity(zip(docids,memberships))
    print 'inv. purity: ', eval.inverse_purity(zip(docids,memberships))
    print '     mi. purity: ', eval.purity(zip(docids,memberships),macro=False)
    print 'mi. inv. purity: ', eval.inverse_purity(zip(docids,memberships),macro=False)

    if 'kmeans' in options.method:
        (memberships,centroids) = kmeans(docvectors, initial=centroids, threshold=options.precision, verbose=options.verbose)
        if options.verbose:
            print ' result centroids: \n', centroids
        if options.output:
            file = open(options.output, 'w')
            for (doc,cluster) in zip(docids,memberships):
                print >> file, doc, cluster
            file.close()
        else:
            print ' result memberships: \n', memberships

        print '%d/%d' % (len(eval.evaluated_docs(zip(docids,memberships))), len(memberships))
        print '     purity: ', eval.purity(zip(docids,memberships))
        print 'inv. purity: ', eval.inverse_purity(zip(docids,memberships))
        print '     mi. purity: ', eval.purity(zip(docids,memberships),macro=False)
        print 'mi. inv. purity: ', eval.inverse_purity(zip(docids,memberships),macro=False,verbose=True)

    # TODO: make it sparse, especially when computing spectral clustering directly from tfidf vectors
    # use scipy.sparse.linalg.eigen

    if 'spectral' in options.method:
        ngmat = ng_matrix_epsilon(centroids,options.epsilon, binary=options.bneighbour)
        degreemat = zeros(ngmat.shape)
        for (i,row) in enumerate(degreemat):
            degreemat[i][i] = sum(ngmat[i])
        laplacian = degreemat - ngmat
        (evals,evecs) = scipy.linalg.eig(laplacian)
        
        vectors = evecs[0:options.sclusters].transpose()
        (mem,cen) = random_kmeans_init(vectors, options.sclusters)
        (mem,cen) = kmeans(vectors, initial=cen, threshold=options.precision, verbose=options.verbose)
        memberships = [mem[x] for x in memberships]
        if options.output:
            file = open(options.output, 'w')
            for (doc,cluster) in zip(docids,memberships):
                print >> file, doc, cluster
            file.close()
        else:
            print ' result memberships: \n', memberships
        print '%d/%d' % (len(eval.evaluated_docs(zip(docids,memberships))), len(memberships))
        print '     purity: ', eval.purity(zip(docids,memberships))
        print 'inv. purity: ', eval.inverse_purity(zip(docids,memberships))
        print '     mi. purity: ', eval.purity(zip(docids,memberships),macro=False)
        print 'mi. inv. purity: ', eval.inverse_purity(zip(docids,memberships),macro=False,verbose=True)
