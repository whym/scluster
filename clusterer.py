#! /usr/bin/env python

import sys
import os
import optparse
import re
import scipy
from scipy import mat, zeros, shape, linalg, diag, dot, random
from math import log
import time

def purity(memberships, references, macro=True, verbose=False):
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

def kmeans(vectors, distance, initial, threshold=1.0E-6, iterations=1000, verbose=False):
    centroids = initial
    memberships = zeros(len(vectors), int)
    old_objective = 0.0
    for it in range(0,iterations):
        objective = 0.0
        # renew memberships
        for (i,v) in enumerate(vectors):
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
        centroids = zeros(centroids.shape, float)
        members = zeros(len(centroids), int)
        for (i,m) in enumerate(memberships):
            centroids[m] += docvectors[i]
            members[m] += 1
        for (i,x) in enumerate(centroids):
            centroids[i] /= members[i]
    
    return memberships, centroids

def docs2vectors_tfidf(dir, vocsize, wpat=re.compile(r'.*')):
    words = {}
    
    tf = []
    df = {}
    docs = os.listdir(dir)
    for file in docs:
        freq = {}
        for line in open(dir + '/' + file):
            for w in re.split(r'\s+', line.strip()):
                x = wpat.match(w)
                if x != None:
                    w = x[1]
                
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
    id2word = sorted(words.keys(), lambda a,b: df[b]-df[a])[0:vocsize]
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
                      dest='method', type=str, default='svd',
                      help='number of words to extract')
    parser.add_option('-d', '--dimension', metavar='DIMENSION',
                      dest='dimension', type=int, default=2000,
                      help='dimension of document vector')
    parser.add_option('-p', '--precision', metavar='PRECISION',
                      dest='precision', type=float, default=1.0E-6,
                      help='dimension of document vector')
    parser.add_option('-r', '--random-seed', metavar='SEED',
                      dest='seed', type=int, default=int(time.time()),
                      help='number of clusters')
    parser.add_option('-K', '--num-clusters', metavar='CLUSTERS',
                      dest='clusters', type=int, default=20,
                      help='number of clusters')
    parser.add_option('-v', '--verbose', metavar='VERBOSE',
                      dest='verbose', action='store_true', default=False,
                      help='turn on verbose message output')
    (options, args) = parser.parse_args()
    (catfile,dir) = tuple(args[0:2])
    if options.verbose:
        print options, catfile, dir
    
    # obtain document vectors
    (docvectors,docids) = docs2vectors_tfidf(dir, options.num_words, re.compile(r'/(.*)'))

    doc2i = {}
    for (i,x) in enumerate(docids):
        doc2i[x] = i
    categories = [[] for x in range(0,len(docvectors))]
    for line in open(catfile):
        v = re.split(r'\s+', line.strip())
        if len(v) >= 2:
            (id,cats) = (v[0], v[1:])
            cats = [x.decode(options.encoding) for x in cats]
            #print id,cats
            if id in doc2i:
                categories[doc2i[id]] = cats
    cmemberships = {}
    for (i,c) in enumerate(categories):
        for x in c:
            cmemberships[i] = x

    if options.method == 'svd':
        # dimensionality reduction by svd
        (U,s,Vh) = linalg.svd(docvectors,full_matrices=False)
        #print shape(U),shape(s),shape(Vh)
        #print [x for x in docvectors[1]]
        docvectors = dot(U[:,:options.dimension], diag(s[:options.dimension]))

        #print [x for x in docvectors[1]]
        if options.verbose:
            print [x for x in s]
    elif options.method == 'spectral':
        pass
    #TODO: compare to random indexing
    
    memberships = [x*options.clusters/len(docvectors) for x in range(0,len(docvectors))]
    random.shuffle(memberships)
    centroids = zeros((options.clusters, options.dimension), float)
    members = zeros(len(centroids), int)
    for (i,m) in enumerate(memberships):
        centroids[m] += docvectors[i]
        members[m] += 1
    for (i,x) in enumerate(centroids):
        centroids[i] /= members[i]
    print 'initial centroids: \n', centroids

    print '     purity: ', purity(list(enumerate(memberships)), categories, verbose=options.verbose)
    print 'inv. purity: ', purity(cmemberships.items(), [[x] for x in memberships], verbose=options.verbose)

    memberships,centroids = kmeans(docvectors, initial=centroids, distance=lambda x,y: linalg.norm(x-y,2), threshold=options.precision, verbose=options.verbose)
    print ' result centroids: \n', centroids
    print ' result memberships: \n', memberships

    print '     purity: ', purity(list(enumerate(memberships)), categories, verbose=options.verbose)
    print 'inv. purity: ', purity(cmemberships.items(), [[x] for x in memberships], verbose=options.verbose)
