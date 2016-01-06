=====================
SCluster
=====================
--------------------------------------------------------
an implementation of spectral clustering for documents
--------------------------------------------------------

 :Homepage: http://github.com/whym/scluster
 :Contact:  http://whym.org

Overview
==============================
Spectral clustering a modern clustering technique considered to be effective for image clustering among others. [#]_ [#]_

This software find clusters among documents based on the bag-of-words representation [#]_ and TF-IDF weighting [#]_.

.. [#] Ulrike von Luxburg, A Tutorial on Spectral Clustering, 2006. http://arxiv.org/abs/0711.0189
.. [#] Chris H. Q. Ding, Spectral Clustering, 2004. http://ranger.uta.edu/~chqding/Spectral/
.. [#] http://en.wikipedia.org/wiki/Bag_of_words_model
.. [#] http://en.wikipedia.org/wiki/Tf%E2%80%93idf

Requirements
==============================
Following softwares are required.

- Python 2.7 or 3.4
- Numpy
- Scipy

How to use
==============================
1. Clone this repository.
2. Prepare documents as raw-text files, and put them in a directory, for example, 'reuters'.
3. Prepare a category file. For example, 'cats.txt' may contain: ::

     14833 palm-oil veg-oil
     14839 ship

   This means that the file '14833' has 'palm-oil' and 'veg-oil' as
   its categories, and '14839' has 'ship' as its category.

4. Run: ``python scluster/clusterer.py cats.txt reusters/ -m kmeans``,

Notes
==============================
- When you use the Reuters set, notice No 17980 might contain
  non-Unicode character at Line 10. It should probably read: "world
  economic growth-side measures ..."

.. [#] http://www.daviddlewis.com/resources/testcollections/reuters21578/
