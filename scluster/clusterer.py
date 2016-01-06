#! /usr/bin/env python

from six.moves import xrange
import sys
import os
import argparse
import re
import codecs
import scipy
import scipy.linalg
import numpy as np
from math import log
import time
from .evaluate import Evaluator


edist = lambda d, cs: np.sum((d - cs) ** 2, axis=1)


def ng_matrix_epsilon(itemvectors, epsilon, distance=edist, binary=False):
    """
    Extract a neighborhood matrix from the given item vectors.

    @param  distance: a function from a vector and a centroid matrix to a distance vector
    @param   epsilon: threshold that defines neighborhood

    """
    c = 0
    matrix = np.zeros((itemvectors.shape[0], itemvectors.shape[0]), float)
    for (i, x) in enumerate(itemvectors):
        matrix[i] = distance(itemvectors[i], itemvectors)
    matrix[matrix > epsilon] = 0.0
    if binary:
        matrix[matrix > 0] = 1.0

    c = np.sum(matrix > 0)
    sys.stderr.write("ng_matrix_epsilon: number of edges = %s (%s)\n" % (c, matrix.shape[0] * matrix.shape[1]))
    return matrix


def kmeans(itemvectors, initial, distance=edist, threshold=1.0E-6, iterations=1000, verbose=False):
    centroids = initial
    memberships = np.zeros(len(itemvectors), int)
    old_objective = 0.0
    for it in xrange(0, iterations):
        # renew memberships
        memberships = np.array([np.argmin(distance(d, centroids)) for d in itemvectors])
        objective = np.sum(np.min(distance(d, centroids)) for d in itemvectors)

        print('iteration %d: %f' % (it, objective))
        if abs((objective - old_objective) / objective) < threshold:
            break
        old_objective = objective

        # renew centroids
        centroids = renew_centroids(itemvectors, memberships, centroids, verbose)

    return memberships, centroids

def renew_centroids(itemvectors, memberships, num_clusters, verbose=False):
    if isinstance(num_clusters, int):
        centroids = np.zeros((num_clusters, itemvectors.shape[1]), float)
    else:
        centroids = num_clusters.copy()
        num_clusters = len(centroids)
    for i in xrange(0, num_clusters):
        members = itemvectors[memberships == i]
        if len(members) > 0:
            centroids[i] = np.mean(members, axis=0)
        else:
            if verbose:
                sys.stderr.write(' cluster %d is empty; not updating\n' % i)
    return centroids

def initialize_random(itemvectors, num_clusters, method='assignment', distance=edist):
    """

    @param  distance: a function from a vector and a centroid matrix to a distance vector
    @param    method: 'assignment' or 'items'- 'assignment': for each item, assign a random label. 'items': choose random k items and let them be centroids, and assign memberships accordingly to other items
    """
    if method == 'assignment':
        memberships = [x * num_clusters // len(itemvectors) for x in xrange(0,len(itemvectors))]
        np.random.shuffle(memberships)

        centroids = renew_centroids(itemvectors, memberships, num_clusters)
    else:
        centroids = itemvectors.copy()
        np.random.shuffle(centroids)
        centroids = centroids[0:num_clusters]
        memberships = np.array([np.argmin(distance(d, centroids)) for d in itemvectors])
    return (memberships, centroids)

def docs2vectors_tfidf(dr, vocsize, verbose=False):
    words = {}
    
    tf = []
    df = {}
    docs = os.listdir(dr)
    for fname in docs:
        freq = {}
        if verbose:
            print('reading ' + fname)
        for line in codecs.open(dr + '/' + fname, encoding='utf-8'):
            for w in re.split(r'\s+', line.strip()):
                words[w] = words.get(w, 0) + 1
                freq[w]  =  freq.get(w, 0) + 1
                #df[w]    =    df.get(w, 0) + 1
        tf.append((fname, freq))
        for w in freq.keys():
            df[w] = df.get(w, 0) + 1
    id2word = sorted(words.keys(), key=lambda x: df[x], reverse=True)[0:vocsize]
    word2id = {}
    mat = np.zeros((len(tf), len(id2word)), float)
    for (i, c) in enumerate(id2word):
        word2id[c] = i
    for (i, row) in enumerate(mat):
        for w in [ x for x in tf[i][1].keys() if x in word2id ]:
            row[word2id[w]] = log(1.0 + tf[i][1][w]) / df[w]
    return (mat, docs)


def print_evaluation(evaluate, docids, memberships):

    def harm_mean(x,y):
        return 2 / ((1/x + 1/y))

    purity   = evaluate.purity(zip(docids, memberships))
    ipurity  = evaluate.inverse_purity(zip(docids, memberships))
    mpurity  = evaluate.purity(zip(docids, memberships), macro=False)
    mipurity = evaluate.inverse_purity(zip(docids, memberships), macro=False, verbose=True)
    print('%d/%d' % (len(evaluate.evaluated_docs(zip(docids, memberships))), len(memberships)))
    print('     purity: %s' % purity)
    print('inv. purity: %s' % ipurity)
    print(' harm. mean: %s' % harm_mean(purity, ipurity))
    print('     mi. purity: %s' % mpurity)
    print('mi. inv. purity: %s' % mipurity)
    print('     harm. mean: %s' % harm_mean(mpurity, mipurity))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoding', metavar='ENCODING',
                        dest='encoding', default='utf-8',
                        help='input/output encoding')
    parser.add_argument('-n', '--num-words', metavar='NUM_WORDS',
                        dest='num_words', type=int, default=5000,
                        help='number of words to extract')
    parser.add_argument('--initialize', choices=['assignment', 'items'],
                        default='assignment')
    parser.add_argument('-m', '--method', metavar='METHOD',
                        dest='method', type=str, default='svd,kmeans',
                        help="""clustering algorithms:
                              comma-separated combination of
                              svd,kmeans,spectral
                        """)
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        dest='output', type=str, default=None,
                        help='output filename')
    parser.add_argument('-a', '--approximation-error', metavar='PERCENT',
                        dest='aerror', type=float, default=1,
                        help='dimension of document vector')
    parser.add_argument('-p', '--precision', metavar='PRECISION',
                        dest='precision', type=float, default=1.0E-4,
                        help='threshold for k-means clustering convergence')
    parser.add_argument('-E', '--epsilon', metavar='EPSILON',
                        dest='epsilon', type=float, default=10.0,
                        help='threshold for constructing neiborhood graph')
    parser.add_argument('-r', '--random-seed', metavar='SEED',
                        dest='seed', type=int, default=int(time.time()),
                        help='random number seed')
    parser.add_argument('-K', '--num-clusters', metavar='CLUSTERS',
                        dest='clusters', type=int, default=60,
                        help='number of clusters')
    parser.add_argument('-S', '--num-spectral-clusters', metavar='SCLUSTERS',
                        dest='sclusters', type=int, default=20,
                        help='number of clusters')
    parser.add_argument('--binary-neighbourhood',
                        dest='bneighbour', action='store_true', default=False,
                        help='use binary value for neighborhood')
    parser.add_argument('-v', '--verbose',
                        dest='verbose', action='store_true', default=False,
                        help='turn on verbose message output')
    parser.add_argument('catfile')
    parser.add_argument('directory')
    args = parser.parse_args()
    args.method = args.method.split(',')
    args.method = dict(zip(args.method, args.method))
    if args.verbose:
        print('%s %s %s' % (args, args.catfile, args.directory))

    np.random.seed(args.seed)

    # obtain document vectors
    (docvectors, docids) = docs2vectors_tfidf(args.directory, args.num_words, args.verbose)

    if args.verbose:
        print('#documents: %d' % len(docvectors))

    evaluate = Evaluator(codecs.open(args.catfile, encoding=args.encoding))

    if 'svd' in args.method:
        # dimensionality reduction by svd
        (U,s,Vh) = scipy.linalg.svd(docvectors, full_matrices=False)
        #print shape(U),shape(s),shape(Vh)
        #print [x for x in docvectors[1]]
        dim = U.shape[0]
        for i in xrange(1, s.shape[0]):
            if 1 - s[0:i].sum()/s.sum() < args.aerror / 100.0:
                dim = i
                break
        docvectors = np.dot(U[:,:dim], np.diag(s[:dim]))
        print('compressed %d dims into %d dims' % (U.shape[0], dim))

        #print [x for x in docvectors[1]]
        if args.verbose:
            print([x for x in s])
    #TODO: compare to random indexing

    (memberships, centroids) = initialize_random(docvectors, args.clusters, method=args.initialize)
    if args.verbose:
        print('initial centroids: \n %s' % centroids)
        print('initial memberships: \n %s' % memberships)

    print_evaluation(evaluate, docids, memberships)

    if 'kmeans' in args.method:
        (memberships, centroids) = kmeans(docvectors, initial=centroids, threshold=args.precision, verbose=args.verbose)
        if args.verbose:
            print(' result centroids: \n %s' % centroids)
        if args.output:
            ofile = open(args.output, 'w')
            for (doc, cluster) in zip(docids, memberships):
                ofile.write('%s %s\n' % (doc, cluster))
            ofile.close()
        else:
            print(' result memberships: \n %s' % memberships)

        print_evaluation(evaluate, docids, memberships)

    # TODO: make it sparse, especially when computing spectral clustering directly from tfidf vectors
    # use scipy.sparse.linalg.eigen

    if 'spectral' in args.method:
        ngmat = ng_matrix_epsilon(centroids, args.epsilon, binary=args.bneighbour)
        degreemat = np.zeros(ngmat.shape)
        for (i, row) in enumerate(degreemat):
            degreemat[i][i] = sum(ngmat[i])
        laplacian = degreemat - ngmat
        (evaluates, evecs) = scipy.linalg.eig(laplacian)

        vectors = evecs[0:args.sclusters].transpose()
        (mem, cen) = initialize_random(vectors, args.sclusters, method=args.initialize)
        (mem, cen) = kmeans(vectors, initial=cen, threshold=args.precision, verbose=args.verbose)
        memberships = [mem[x] for x in memberships]
        if args.output:
            ofile = open(args.output, 'w')
            for (doc, cluster) in zip(docids, memberships):
                ofile.write('%s %s\n' % (doc, cluster))
            ofile.close()
        else:
            print(' result memberships: \n %s' % memberships)
        print_evaluation(evaluate, docids, memberships)
