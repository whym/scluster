#! /usr/bin/env python

import sys
import os
import argparse
import re
import codecs
import scipy
import scipy.linalg
from scipy import mat, zeros, shape, diag, dot, random
from math import log
import time
from evaluate import Evaluator

def ng_matrix_epsilon(itemvectors, epsilon, distance=lambda x,y: scipy.linalg.norm(x-y,2), binary=False):
    """
    Extract a neighborhood matrix from the given item vectors.

    @param  distance: a function two vectors (x,y) to a double value
    @param   epsilon: threshold that defines neighborhood

    """
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
    sys.stderr.write("ng_matrix_epsilon: number of edges = %s\n" % c)
    return matrix

def kmeans(itemvectors, initial, distance=lambda x,y: scipy.linalg.norm(x-y,2), threshold=1.0E-6, iterations=1000, verbose=False):
    centroids = initial
    memberships = zeros(len(itemvectors), int)
    old_objective = 0.0
    for it in range(0,iterations):
        objective = 0.0
        # renew memberships
        for (i,v) in enumerate(itemvectors):
            mi = (float('inf'),None)
            for (j,c) in enumerate(centroids):
                d = distance(v,c)
                if d < mi[0]:
                    mi = (d, j)
                    memberships[i] = j
            objective += mi[0]
        if verbose:
            print(objective)
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
                sys.stderr.write('''%s %s
 cluster %d is empty
''' % (members, memberships, i))
                centroids[i] = oldc[i]
            else:
                centroids[i] /= members[i]
    
    return memberships, centroids

def random_kmeans_init(itemvectors, num_clusters):
    """
    for each item, assign a random label
    """
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

def random_kmeans_init2(itemvectors, num_clusters, distance=lambda x,y: scipy.linalg.norm(x-y,2)):
    """
    choose random k items and let them be centroids,
    and assign memberships accordingly to other items
    """
    centroids = itemvectors.copy()
    random.shuffle(centroids)
    centroids = centroids[0:num_clusters]

    memberships = [0 for x in itemvectors]
    for (i,v) in enumerate(itemvectors):
        mi = (float('inf'),None)
        for (j,c) in enumerate(centroids):
            d = distance(v,c)
            if d < mi[0]:
                mi = (d, j)
                memberships[i] = j
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
    mat = zeros((len(tf), len(id2word)), float)
    for (i,c) in enumerate(id2word):
        word2id[c] = i
    for (i,row) in enumerate(mat):
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
                        dest='precision', type=float, default=1.0E-6,
                        help='threshold for k-means clustering convergence')
    parser.add_argument('-E', '--epsilon', metavar='EPSILON',
                        dest='epsilon', type=float, default=10.0,
                        help='threshold for constructing neiborhood graph')
    parser.add_argument('-r', '--random-seed', metavar='SEED',
                        dest='seed', type=int, default=int(time.time()),
                        help='random number seed')
    parser.add_argument('-K', '--num-clusters', metavar='CLUSTERS',
                        dest='clusters', type=int, default=40,
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
    
    random.seed(args.seed)

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
        docvectors = dot(U[:,:dim], diag(s[:dim]))
        print('comopressed %d dims into %d dims' % (U.shape[0], dim))

        #print [x for x in docvectors[1]]
        if args.verbose:
            print([x for x in s])
    #TODO: compare to random indexing
    
    (memberships, centroids) = random_kmeans_init2(docvectors, args.clusters)
    print('initial centroids: \n %s' % centroids)
    print('initial memberships: \n %s' % memberships)

    print_evaluation(evaluate, docids, memberships)

    if 'kmeans' in args.method:
        (memberships,centroids) = kmeans(docvectors, initial=centroids, threshold=args.precision, verbose=args.verbose)
        if args.verbose:
            print(' result centroids: \n %s' % centroids)
        if args.output:
            ofile = open(args.output, 'w')
            for (doc,cluster) in zip(docids,memberships):
                ofile.write('%s %s\n' % (doc, cluster))
            ofile.close()
        else:
            print(' result memberships: \n %s' % memberships)

        print_evaluation(evaluate, docids, memberships)

    # TODO: make it sparse, especially when computing spectral clustering directly from tfidf vectors
    # use scipy.sparse.linalg.eigen

    if 'spectral' in args.method:
        ngmat = ng_matrix_epsilon(centroids,args.epsilon, binary=args.bneighbour)
        degreemat = zeros(ngmat.shape)
        for (i,row) in enumerate(degreemat):
            degreemat[i][i] = sum(ngmat[i])
        laplacian = degreemat - ngmat
        (evaluates, evecs) = scipy.linalg.eig(laplacian)
        
        vectors = evecs[0:args.sclusters].transpose()
        (mem,cen) = random_kmeans_init2(vectors, args.sclusters)
        (mem,cen) = kmeans(vectors, initial=cen, threshold=args.precision, verbose=args.verbose)
        memberships = [mem[x] for x in memberships]
        if args.output:
            ofile = open(args.output, 'w')
            for (doc,cluster) in zip(docids,memberships):
                ofile.write('%s %s\n' % (doc, cluster))
            ofile.close()
        else:
            print(' result memberships: \n %s' % memberships)
        print_evaluation(evaluate, docids, memberships)
