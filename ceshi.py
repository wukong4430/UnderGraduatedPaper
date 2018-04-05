# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-03-28 14:24:06
# @Last Modified by:   Kicc Shen
# @Last Modified time: 2018-04-01 21:12:20

import numpy as np
from itertools import combinations

# for i in range(2):
#     column = b[:, i]
#     print(column)
#     MAX = np.max(column)
#     MIN = np.min(column)

#     column = (column - MIN) / (MAX - MIN)
#     c_b[:, i] = column

# print(c_b)

# dist = []
# for (i, j) in combinations(range(3), 2):
#     euclidean_dist = np.linalg.norm(b[i, :] - b[j, :])
#     dist.append(euclidean_dist)


def select(X_src, X_tar):
    n1 = X_src.shape[0]
    n2 = X_tar.shape[0]

    def help(n_samples, X):
        # calc DIST. DIST = {dij:}
        dist = []
        for (i, j) in combinations(range(n_samples), 2):
            euclidean_dist = np.linalg.norm(X[i, :] - X[j, :])
            dist.append(euclidean_dist)

        dcv = {}
        dcv['dist_mean'] = np.mean(dist)
        dcv['dist_median'] = np.median(dist)
        dcv['dist_min'] = np.min(dist)
        dcv['dist_max'] = np.max(dist)
        dcv['dist_std'] = np.std(dist)
        dcv['numInstances'] = n_samples

        return dcv

    dcv_src = help(n_samples=n1, X=X_src)
    dcv_tar = help(n_samples=n2, X=X_tar)
    print('DCV of src :', dcv_src)
    print('DCV of tar :', dcv_tar)


if __name__ == '__main__':

    b = np.array([[10, 2], [7, 12], [2, 9]])
    a = np.array([[1, 2], [3, 5], [5, 8]])
    c_b = np.zeros([3, 2])

    select(a, b)
