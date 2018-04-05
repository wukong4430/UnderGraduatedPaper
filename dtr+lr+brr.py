from sklearn import linear_model
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from rankSVM import RankSVM
from PerformanceMeasure import PerformanceMeasure
from Processing import Processing
import pandas as pd


def pred_result(training_data_X, training_data_y, test_data_X):
    '''

    return: 7个回归模型对test_data_X的预测值，预测值会四舍五入取整
            stored in a list.

    '''

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)

    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)

    bayes = BayesianRidge().fit(training_data_X, training_data_y)

    mlpr = MLPRegressor().fit(training_data_X, training_data_y)

    svr = SVR().fit(training_data_X, training_data_y)

    knr = KNeighborsRegressor().fit(training_data_X, training_data_y)

    gbr = GradientBoostingRegressor().fit(training_data_X, training_data_y)

    rs = RankSVM(C=1.0).fit(training_data_X, training_data_y)
    # numpy.around() 四舍五入

    return [(np.around(dtr.predict(test_data_X)), 'DecisionTreeRegressor'), (np.around(lr.predict(test_data_X)), 'LinearRegression'), (np.around(bayes.predict(test_data_X)), 'BayesianRidge'),
            (np.around(mlpr.predict(test_data_X)), 'MLPRegressor'), (np.around(
                svr.predict(test_data_X)), 'SVR'), (np.around(knr.predict(test_data_X)), 'KNeighborsRegressor'),
            (np.around(gbr.predict(test_data_X)), 'GradientBoostingRegressor'), (np.around(gbr.predict2(test_data_X)), 'RankingSVM')]


def bootstrap():
    '''同类数据集共同测评

    '''
    dataset = Processing().import_data()

    training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
    ).separate_data(dataset)

    y_pred = pred_result(training_data_X, training_data_y, testing_data_X)

    for algorithm, name in y_pred:

        fpa = PerformanceMeasure(testing_data_y, algorithm).FPA()

        # aee_result = PerformanceMeasure(testing_data_y, i).AEE()

        print('FPA for %s is %s', (name, fpa))

        # print('AEE_result', aee_result)


def bootstrap_single_data():
    '''单独测评每个数据集

    '''
    algorithm_name_list = []
    filename_list = []
    average_fpa_list = []

    for dataset, filename in Processing().import_single_data():

        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        y_pred = pred_result(training_data_X, training_data_y, testing_data_X)

        for algorithm, name in y_pred:

            sum_fpa = 0.0
            for _ in range(100):

                fpa = PerformanceMeasure(testing_data_y, algorithm).FPA()

                sum_fpa += fpa

            average_fpa = sum_fpa / 100

            print('Average fpa of algorithm: 【%s】 for dataset: 【%s】 is %s.' %
                  (name, filename, average_fpa))

            algorithm_name_list.append(name)
            filename_list.append(filename)
            average_fpa_list.append(average_fpa)

    dataframe = pd.DataFrame(
        {'algorithm': algorithm_name_list, 'dataset': filename_list, 'fpa': average_fpa_list})
    dataframe.to_csv("fpa-result.csv", index=False, sep=',')


def cross_validation_single_data(classifier='dtr'):

    for dataset, filename in Processing().import_single_data():

        training_data_list, testing_data_list = Processing().cross_validation(dataset)

        # print(len(training_data_list))

        testing_data_y = np.array(None)
        classifier_pred_y = np.array(None)

        for training_data_i, testing_data_i in zip(training_data_list, testing_data_list):

            # print(training_data_i.shape)

            training_data_i_X, training_data_i_y = training_data_i[:,
                                                                   :-1], training_data_i[:, -1:]
            # print(training_data_i_X.shape)
            # print(training_data_i_y.shape)
            testing_data_i_X, testing_data_i_y = testing_data_i[:, :-
                                                                1], testing_data_i[:, -1:]

            # testing_data_y.append(testing_data_i_y)
            testing_data_y = np.append(testing_data_y, testing_data_i_y)

            if classifier == 'dtr':
                dtr = DecisionTreeRegressor().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'lr':
                dtr = linear_model.LinearRegression().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'bayes':
                dtr = BayesianRidge().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'mlpr':
                dtr = MLPRegressor().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'svr':
                dtr = SVR().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'knr':
                dtr = KNeighborsRegressor().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

            if classifier == 'gbr':
                dtr = GradientBoostingRegressor().fit(training_data_i_X, training_data_i_y)
                # print('this shape =', dtr.predict(testing_data_i_X).shape)
                classifier_pred_y = np.append(
                    classifier_pred_y, np.around(dtr.predict(testing_data_i_X)))

        testing_data_y = testing_data_y[1:]
        classifier_pred_y = classifier_pred_y[1:]

        # print('shape1 =', testing_data_y.shape)
        # print('shape2 =', classifier_pred_y.shape)

        classifier_fpa = PerformanceMeasure(
            testing_data_y, classifier_pred_y).FPA()
        print('Fpa of algorithm: 【%s】 for dataset: 【%s】 is %s.' %
              (classifier, filename, classifier_fpa))


if __name__ == '__main__':

    bootstrap()

    # bootstrap_single_data()

    # 10-fold
    # classifiers = ['dtr', 'lr', 'bayes', 'mlpr', 'svr', 'knr', 'gbr']
    # for classifier in classifiers:

    #     cross_validation_single_data(classifier=classifier)
