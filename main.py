# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-04-22 17:15:01
# @Last Modified by:   KICC
# @Last Modified time: 2018-04-30 09:25:06


from rankSVM import RankSVM
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure
from Processing import Processing
import pandas as pd
import numpy as np
from tca import TCA
from TCA_Plus import TCA_PLUS
from nnfilter import NN_filter
from peterfilter import peter_filter


def do_nothing(classifier, training_data_X, training_data_y, testing_data_X, testing_data_y):
    ############################################################
    # DO NOTHING
    rs_nothing = classifier.fit(training_data_X, training_data_y)
    # camel as test/target
    rs_nothing_pred_y = rs_nothing.predict2(testing_data_X)
    rs_nothing_fpa = PerformanceMeasure(
        testing_data_y, rs_nothing_pred_y).FPA()
    # print('rs_nothing_fpa :', rs_nothing_fpa)

    # print()
    aver_nothing_list.append(rs_nothing_fpa)


def do_tca(classifier, training_data_X, training_data_y, testing_data_X, testing_data_y):
    ##########################################################
    # TCA
    tca = TCA(dim=8, kerneltype='rbf', kernelparam=1.5, mu=0.0035)
    training_tca_X, testing_tca_X, x_tar_o_tca = tca.fit_transform(
        training_data_X, testing_data_X)

    training_tca_y = training_data_y
    testing_tca_y = testing_data_y
    # ant as train/source
    rs_tca = classifier.fit(training_tca_X, training_tca_y)
    # camel as test/target
    rs_tca_pred_y = rs_tca.predict2(testing_tca_X)
    rs_tca_fpa = PerformanceMeasure(
        testing_tca_y, rs_tca_pred_y).FPA()
    # print('rs_tca_fpa :', rs_tca_fpa)

    # print()
    aver_tca_list.append(rs_tca_fpa)


def do_tcaplus(classifier, training_data_X, training_data_y, testing_data_X, testing_data_y):
    ##########################################################
    # TCA_PLUS
    tcap = TCA_PLUS(X_src=training_data_X, X_tar=testing_data_X)
    training_tcap_X, testing_tcap_X = tcap.transform(
        dim=8, kerneltype='rbf', kernelparam=1.5, mu=0.0035)

    training_tcap_y = training_data_y
    testing_tcap_y = testing_data_y
    # ant as train/source
    rs_tcap = classifier.fit(training_tcap_X, training_tcap_y)
    # camel as test/target
    rs_tcap_pred_y = rs_tcap.predict2(testing_tcap_X)
    rs_tcap_fpa = PerformanceMeasure(
        testing_tcap_y, rs_tcap_pred_y).FPA()
    # print('rs_tcap_fpa :', rs_tcap_fpa)

    # print()
    aver_tcap_list.append(rs_tcap_fpa)


def do_nnfilter(classifier, training_data_X, training_data_y, testing_data_X, testing_data_y):
    ##########################################################
    # NN FILTER
    nn = NN_filter(training_data_X, training_data_X,
                   testing_data_X, testing_data_y)

    training_nn_X, training_nn_y, testing_nn_X, testing_nn_y = nn.transform()

    rs_nn_filter = classifier.fit(training_nn_X, training_nn_y)
    # camel as test/target
    rs_nn_pred_y = rs_nn_filter.predict2(testing_nn_X)
    rs_nn_fpa = PerformanceMeasure(testing_nn_y, rs_nn_pred_y).FPA()
    # print('rs_nn_fpa :', rs_nn_fpa)

    aver_nn_list.append(rs_nn_fpa)
    return training_nn_X, training_nn_y, testing_nn_X, testing_nn_y


def do_peterfilter(classifier, training_data_X, training_data_y, testing_data_X, testing_data_y):
    ##########################################################
    # Peter FILTER
    peter = peter_filter(training_data_X, training_data_X,
                         testing_data_X, testing_data_y)

    training_peter_X, training_peter_y, testing_peter_X, testing_peter_y = peter.transform()

    rs_peter_filter = classifier.fit(training_peter_X, training_peter_y)
    # camel as test/target
    rs_peter_pred_y = rs_peter_filter.predict2(testing_peter_X)
    rs_peter_fpa = PerformanceMeasure(testing_peter_y, rs_peter_pred_y).FPA()
    # print('rs_peter_fpa :', rs_peter_fpa)
    aver_peter_list.append(rs_peter_fpa)
    return training_peter_X, training_peter_y, testing_peter_X, testing_peter_y


def main(classifier):
    """
    1.从原始数据集中获取数据
    2.数据集用tca/nn filter处理，得到特征选择后的数据
    3.将处理后的数据导入Ranking svm
    4.除了Ranking svm，其他的分类器可以自己加

    example: take ant as a source project and camel as a target project.
             apply tca to transform ant(as train) and camel(as test).

    """

    src_tar_list = [('ant', 'jedit'), ('ant', 'poi'), ('ant', 'xalan'), ('camel', 'log4j'),
                    ('camel', 'synapse'), ('camel', 'xerces'), ('ivy',
                                                                'lucene'), ('ivy', 'velocity'),
                    ('ivy', 'ant'), ('jedit', 'poi'), ('jedit',
                                                       'xalan'), ('jedit', 'camel'),
                    ('log4j', 'synapse'), ('log4j', 'xerces'), ('log4j',
                                                                'ivy'), ('lucene', 'velocity'),
                    ('lucene', 'ant'), ('lucene',
                                        'jedit'), ('poi', 'xalan'), ('poi', 'camel'),
                    ('poi', 'log4j'), ('synapse', 'xerces'), ('synapse',
                                                              'ivy'), ('synapse', 'lucene'),
                    ('velocity', 'ant'), ('velocity',
                                          'jedit'), ('velocity', 'poi'), ('xalan', 'camel'),
                    ('xalan', 'log4j'), ('xalan', 'synapse'), ('xerces', 'ivy'), ('xerces', 'lucene'), ('xerces', 'velocity')]
    for _ in range(5):
        # Nothing needs an average value.
        for index, unit in enumerate(src_tar_list):
            src = unit[0]
            tar = unit[1]
            # print('-' * 20)
            # print('src =', src)
            # print('tar =', tar)
            dataset_src = Processing(folder_name=src).import_data()
            dataset_tar = Processing(folder_name=tar).import_data()

            training_data_X, training_data_y = Processing(
                folder_name=src).transfrom_data(dataset_src)

            testing_data_X, testing_data_y = Processing(
                folder_name=tar).transfrom_data(dataset_tar)

            # nothing
            do_nothing(classifier=classifier, training_data_X=training_data_X,
                       training_data_y=training_data_y, testing_data_X=testing_data_X, testing_data_y=testing_data_y)

    for index, unit in enumerate(src_tar_list):
        src = unit[0]
        tar = unit[1]
        dataset_src = Processing(folder_name=src).import_data()
        dataset_tar = Processing(folder_name=tar).import_data()

        training_data_X, training_data_y = Processing(
            folder_name=src).transfrom_data(dataset_src)

        testing_data_X, testing_data_y = Processing(
            folder_name=tar).transfrom_data(dataset_tar)

        # tca
        do_tca(classifier=classifier, training_data_X=training_data_X,
               training_data_y=training_data_y, testing_data_X=testing_data_X, testing_data_y=testing_data_y)

        # tcap
        do_tcaplus(classifier=classifier, training_data_X=training_data_X,
                   training_data_y=training_data_y, testing_data_X=testing_data_X, testing_data_y=testing_data_y)

        # nn
        training_nn_X, training_nn_y, testing_nn_X, testing_nn_y = do_nnfilter(classifier=classifier, training_data_X=training_data_X,
                                                                               training_data_y=training_data_y, testing_data_X=testing_data_X, testing_data_y=testing_data_y)

        # peter
        training_peter_X, training_peter_y, testing_peter_X, testing_peter_y = do_peterfilter(classifier=classifier, training_data_X=training_data_X,
                                                                                              training_data_y=training_data_y, testing_data_X=testing_data_X, testing_data_y=testing_data_y)

        # tca+nn

        # tca+peter

        # tcap+nn

        # tcap+peter


def deal_nothing(nothing_list):
    # 将120长度的nothinglist转化成24长度
    count = 0
    sums = 0
    res = []
    for i in nothing_list:
        sums += i
        if count % 5 == 4:
            res.append(sums)
            sums = 0
        count += 1
    res = [i / 5 for i in res]
    return res


if __name__ == '__main__':
    rs = RankSVM()
    lr = LinearRegression()
    dtr = DecisionTreeRegressor()
    aver_nothing_list = []

    aver_tca_list = []

    aver_tcap_list = []

    aver_nn_list = []

    aver_peter_list = []

    main(classifier=rs)

    new_nothing_list = deal_nothing(aver_nothing_list)
    for index in range(33):
        print('unit {0}, nothing={1}, tca={2}, tcap={3}, nn={4}, peter={5}'.format(
            index, new_nothing_list[index + 1], aver_tca_list[index],
            aver_tcap_list[index], aver_nn_list[index], aver_peter_list[index]))

    # print('\nunit 1 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_1) / 10)
    # print('aver of tca :', sum(aver_tca_1) / 10)
    # print('aver of tcap :', sum(aver_tcap_1) / 10)
    # print('aver of nn :', sum(aver_nn_1) / 10)
    # print('aver of peter :', sum(aver_peter_1) / 10)

    # print('\nunit 2 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_2) / 10)
    # print('aver of tca :', sum(aver_tca_2) / 10)
    # print('aver of tcap :', sum(aver_tcap_2) / 10)
    # print('aver of nn :', sum(aver_nn_2) / 10)
    # print('aver of peter :', sum(aver_peter_2) / 10)

    # print('\nunit 3 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_3) / 10)
    # print('aver of tca :', sum(aver_tca_3) / 10)
    # print('aver of tcap :', sum(aver_tcap_3) / 10)
    # print('aver of nn :', sum(aver_nn_3) / 10)
    # print('aver of peter :', sum(aver_peter_3) / 10)

    # print('\nunit 4 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_4) / 10)
    # print('aver of tca :', sum(aver_tca_4) / 10)
    # print('aver of tcap :', sum(aver_tcap_4) / 10)
    # print('aver of nn :', sum(aver_nn_4) / 10)
    # print('aver of peter :', sum(aver_peter_4) / 10)

    # print('\nunit 5 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_5) / 10)
    # print('aver of tca :', sum(aver_tca_5) / 10)
    # print('aver of tcap :', sum(aver_tcap_5) / 10)
    # print('aver of nn :', sum(aver_nn_5) / 10)
    # print('aver of peter :', sum(aver_peter_5) / 10)

    # print('\nunit 6 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_6) / 10)
    # print('aver of tca :', sum(aver_tca_6) / 10)
    # print('aver of tcap :', sum(aver_tcap_6) / 10)
    # print('aver of nn :', sum(aver_nn_6) / 10)
    # print('aver of peter :', sum(aver_peter_6) / 10)

    # print('\nunit 7 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_7) / 10)
    # print('aver of tca :', sum(aver_tca_7) / 10)
    # print('aver of tcap :', sum(aver_tcap_7) / 10)
    # print('aver of nn :', sum(aver_nn_7) / 10)
    # print('aver of peter :', sum(aver_peter_7) / 10)

    # print('\nunit 8 :')
    # print('-' * 40)
    # print('aver of nothing :', sum(aver_nothing_8) / 10)
    # print('aver of tca :', sum(aver_tca_8) / 10)
    # print('aver of tcap :', sum(aver_tcap_8) / 10)
    # print('aver of nn :', sum(aver_nn_8) / 10)
    # print('aver of peter :', sum(aver_peter_8) / 10)
