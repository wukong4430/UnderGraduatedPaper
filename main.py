# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-04-22 17:15:01
# @Last Modified by:   Kicc Shen
# @Last Modified time: 2018-04-24 22:40:24


from rankSVM import RankSVM
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure
from Processing import Processing
import pandas as pd
import numpy as np
from TCA_Plus import TCA_PLUS
from nnfilter import NN_filter


def main(classifier):
    """
    1.从原始数据集中获取数据
    2.数据集用tca/nn filter处理，得到特征选择后的数据
    3.将处理后的数据导入Ranking svm
    4.除了Ranking svm，其他的分类器可以自己加

    example: take ant as a source project and camel as a target project.
             apply tca to transform ant(as train) and camel(as test).

    """
    dataset_ant = Processing(folder_name='ant').import_data()
    dataset_camel = Processing(folder_name='camel').import_data()

    training_data_X, training_data_y = Processing(
        folder_name='ant').transfrom_data(dataset_ant)

    testing_data_X, testing_data_y = Processing(
        folder_name='camel').transfrom_data(dataset_camel)

    ############################################################
    # DO NOTHING
    rs_nothing = classifier.fit(training_data_X, training_data_y)
    # camel as test/target
    rs_nothing_pred_y = rs_nothing.predict2(testing_data_X)
    rs_nothing_fpa = PerformanceMeasure(
        testing_data_y, rs_nothing_pred_y).FPA()
    print('rs_nothing_fpa :', rs_nothing_fpa)

    ##########################################################
    # TCA_PLUS
    tcap = TCA_PLUS(X_src=training_data_X, X_tar=testing_data_X)
    training_tca_X, testing_tca_X = tcap.transform(
        dim=8, kerneltype='rbf', kernelparam=1.5, mu=0.0035)

    training_tca_y = training_data_y
    testing_tca_y = testing_data_y
    # ant as train/source
    rs_tca = classifier.fit(training_tca_X, training_tca_y)
    # camel as test/target
    rs_tca_pred_y = rs_tca.predict2(testing_tca_X)
    rs_tca_fpa = PerformanceMeasure(
        testing_tca_y, rs_tca_pred_y).FPA()
    print('rs_tca_fpa :', rs_tca_fpa)
    print()

    ##########################################################
    # NN FILTER
    nn = NN_filter(training_data_X, training_data_X,
                   testing_data_X, testing_data_y)

    training_nn_X, training_nn_y, testing_nn_X, testing_nn_y = nn.transform()

    rs_nn_filter = classifier.fit(training_nn_X, training_nn_y)
    # camel as test/target
    rs_nn_pred_y = rs_nn_filter.predict2(testing_nn_X)
    rs_nn_fpa = PerformanceMeasure(testing_nn_y, rs_nn_pred_y).FPA()
    print('rs_nn_fpa :', rs_nn_fpa)


if __name__ == '__main__':
    rs = RankSVM()
    lr = LinearRegression()
    dtr = DecisionTreeRegressor()
    main(classifier=rs)
