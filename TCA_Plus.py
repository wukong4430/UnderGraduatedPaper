# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-03-28 13:24:42
# @Last Modified by:   kicc
# @Last Modified time: 2018-04-03 19:24:05

import numpy as np
from itertools import combinations
from tca import TCA


class TCA_PLUS:

    def __init__(self, X_src, X_tar):
        self.X_src = X_src
        self.X_tar = X_tar

    def normalization(self, normal_type='NoN'):
    """ Normalization on the original data before execute TCA.
    param: self.X_src matrix shape like (n_samples, m_features)
    param: self.X_tar matrix shape like (n_samples, m_features)

    return: X after normalization.
    """

    n_samples_src, m_features_src = self.X_src.shape
    n_samples_tar, m_features_tar = self.X_tar.shape

    normaled_X_src = np.zeros([n_samples_src, m_features_src])
    normaled_X_tar = np.zeros([n_samples_tar, m_features_tar])

    if normal_type == 'NoN':
        '''No normalization is applied.'''
        pass

    elif normal_type == 'N1':
        '''Both for source and target projects.'''
        for i in range(m_features_src):
            column_src = self.X_src[:, i]
            MAX = np.max(column_src)
            MIN = np.min(column_src)

            new_column = (column_src - MIN) / (MAX - MIN)
            normaled_self.X_src[:, i] = new_column

        for i in range(m_features_tar):
            column_tar = self.X_tar[:, i]
            MAX = np.max(column_tar)
            MIN = np.min(column_tar)

            new_column = (column_tar - MIN) / (MAX - MIN)
            normaled_self.X_tar[:, i] = new_column

    elif normal_type == 'N2':
        '''Both for source and target projects.'''
        for i in range(m_features_src):
            column_src = self.X_src[:, i]
            STD = np.std(column_src)
            MEAN = np.mean(column_src)

            new_column = (column_src - MEAN) / STD
            normaled_self.X_src[:, i] = new_column

        for i in range(m_features_tar):
            column_tar = self.X_tar[:, i]
            STD = np.std(column_tar)
            MEAN = np.mean(column_tar)

            new_column = (column_tar - MEAN) / STD
            normaled_self.X_tar[:, i] = new_column

    elif normal_type == 'N3':
        '''Only for source projects.'''
        for i in range(m_features_src):
            column_src = self.X_src[:, i]
            MEAN = np.mean(column_src)
            STD = np.std(column_src)

            new_column = (column_src - MEAN) / STD
            normaled_self.X_src[:, i] = new_column

    elif normal_type == 'N4':
        '''Only for target projects.'''
        for i in range(m_features_tar):
            column_tar = self.X_tar[:, i]
            STD = np.std(column_tar)
            MEAN = np.mean(column_tar)

            new_column = (column_tar - MEAN) / STD
            normaled_self.X_tar[:, i] = new_column

    else:
        pass

    return normaled_X_src, normaled_X_tar

    def selection(self):
        '''Input: Similarity vector Ss=>t, where s and t are source and target projects.

            Output: Normalization option(NoN|N1|N2|N3|N4)

        '''

        # calc 6 elements for X.
        n_samples_src, m_features_src = self.X_src.shape
        n_samples_tar, m_features_tar = self.X_tar.shape

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

        dcv_src = help(n_samples=n_samples_src, X=self.X_src)
        dcv_tar = help(n_samples=n_samples_tar, X=self.X_tar)

        # Rule1
        if dcv_src['dist_mean'] * 0.9 <= dcv_tar['dist_mean'] and
            dcv_src['dist_mean'] * 1.1 >= dcv_tar['dist_mean'] and
            dcv_src['dist_std'] * 0.9 <= dcv_tar['dist_std'] and
            dcv_src['dist_std'] * 1.1 >= dcv_tar['dist_std']:
            normal_selection = 'NoN'

        # Rule2
        elif (dcv_src['numInstances'] * 1.6 < dcv_tar['numInstances'] and
              dcv_src['dist_min'] * 1.6 < dcv_tar['dist_min'] and
              dcv_src['dist_max'] * 1.6 < dcv_tar['dist_max'])
            or
            (dcv_src['numInstances'] * 0.4 > dcv_tar['numInstances'] and
             dcv_src['dist_min'] * 0.4 > dcv_tar['dist_min'] and
             dcv_src['dist_max'] * 0.4 > dcv_tar['dist_max']):
            normal_selection = 'N1'

        # Rule3
        elif (dcv_src['dist_std'] * 1.6 < dcv_tar['dist_std'] and
              dcv_src['numInstances'] > dcv_tar['numInstances']) or
            (dcv_src['dist_std'] * 0.4 > dcv_tar['dist_std'] and
             dcv_src['numInstances'] < dcv_tar['numInstances']):
            normal_selection = 'N3'

        # Rule4
        elif (dcv_src['dist_std'] * 1.6 < dcv_tar['dist_std'] and
              dcv_src['numInstances'] < dcv_tar['numInstances']) or
            (dcv_src['dist_std'] * 0.4 > dcv_tar['dist_std'] and
             dcv_src['numInstances'] > dcv_tar['numInstances']):
            normal_selection = 'N4'

        else:
            normal_selection = 'N2'

        return normal_selection

    def transform(self):
        """Call TCA """
        my_tca = TCA(dim=30, kerneltype='rbf', kernelparam=1, mu=1)
        x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(
            self.X_src, X_tar)
