# -*- coding: utf-8 -*-

"""
Copyright (c) 2016 Randal S. Olson
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from sklearn.neighbors import KDTree


class ReliefF(object):
    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        """Sets up ReliefF to perform feature selection.
        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.
        Returns
        -------
        None
        """

        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep

    def fit(self, X, y):
        """Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        None
        """
        self.feature_scores = np.zeros(X.shape[1])
        self.tree = KDTree(X)

        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(X[source_index].reshape(1, -1), k=self.n_neighbors + 1)

            # First match is self, so ignore it
            for neighbor_index in indices[0][1:]:
                similar_features = X[source_index] == X[neighbor_index]
                label_match = y[source_index] == y[neighbor_index]

                # If the labels match, then increment features that match and decrement features that do not match
                # Do the opposite if the labels do not match
                if label_match:
                    self.feature_scores[similar_features] += 1.
                    self.feature_scores[~similar_features] -= 1.
                else:
                    self.feature_scores[~similar_features] += 1.
                    self.feature_scores[similar_features] -= 1.

        self.top_features = np.argsort(self.feature_scores)[::-1]

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[self.n_features_to_keep]]


mawilab_label = np.load("./data/mawilab_label_10w.npy").astype(np.longlong)
mawilab_data_all = np.load("./data/mawilab_10w.npy")
slides = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
          [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
          [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
          [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
          [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
          [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
           103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
           116, 117, 118, 119, 120, 121, 122, 123, 124],
          [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
           138, 139],
          [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
           153, 154],
          [155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
           168, 169],
          [170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
           183, 184],
          [185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
           198, 199],
          [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
           213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
           226, 227, 228, 229, 230, 231, 232, 233, 234]
          ]
lx = []
for i in range(len(slides)):
    lx.append(mawilab_data_all[:, slides[i]].mean(axis=1))
p = np.stack(lx)
r = ReliefF(n_neighbors=6)
r.fit(p, mawilab_label)
