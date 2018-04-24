# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-04-23 12:40:44
# @Last Modified by:   Kicc Shen
# @Last Modified time: 2018-04-23 13:59:17

from Processing import Processing
import numpy as np


class NN_filter:

    def __init__(self, X_src, y_src, X_tar, y_tar):
        self.X_src = X_src
        self.y_src = y_src
        self.X_tar = X_tar
        self.y_tar = y_tar

    def get_distance(self, test_data, train_data):
        # test_data: a vector
        # train_data: a vector
        # 两个向量的欧式距离
        distance = 0.0
        for index in range(len(test_data)):
            distance = distance + \
                np.math.pow(test_data[index] - train_data[index], 2)
        return np.math.sqrt(distance)

    def insertsort(self, mylist):
        # 按照mylist中的distance升序排序
        for i in range(1, len(mylist)):

            if mylist[i - 1][0] > mylist[i][0]:
                tmp = mylist[i]
                j = i
                while j > 0 and mylist[j - 1][0] > tmp[0]:
                    mylist[j] = mylist[j - 1]
                    j -= 1
                mylist[j] = tmp

    def equal(self, list1, list2):
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                return False
        return True

    def transform(self):
        # x_train matrix; y_train vector
        # x_test matrix; y_test vector
        X_train, y_train = self.X_src, self.y_src
        X_test, y_test = self.X_tar, self.y_tar

        X_result = []  # 保存的是 train_vector_x
        y_result = []
        # print(test_name)
        for test_data in X_test:
            # for each vector (a row)
            mlist = []  # 每个元素是一个list, 包含distance, train_vector_x, train_label_y
            for index in range(len(X_train)):
                # go through all x from x1~xn
                distance = self.get_distance(test_data, X_train[index])
                if len(mlist) >= 10:
                    temp = mlist.pop()
                    if distance < temp[0]:  # 当前的distance与第十个distance比较。选更近的。
                        mlist.append(
                            [distance, X_train[index], y_train[index]])
                        self.insertsort(mlist)   # 按照distance升序排序
                    else:
                        mlist.append(temp)  # 如果当前的distance还不如原来的近， 那就不做任何改变
                else:
                    # 还没有达到十个之前，直接加入，同时也要排序
                    mlist.append([distance, X_train[index], y_train[index]])
                    self.insertsort(mlist)
            for item in mlist:
                flag = True
                for item2 in X_result:
                    if self.equal(item[1], item2):
                        flag = False
                        break
                if flag:
                    # result里还没有就添加，有就跳过
                    X_result.append(item[1])
                    y_result.append(item[2])
                else:
                    break
        X_result = np.asarray(X_result)
        y_result = np.asarray(y_result)
        return X_result, y_result, X_test, y_test


if __name__ == '__main__':
    dataset_ant = Processing(folder_name='ant').import_data()
    dataset_camel = Processing(folder_name='camel').import_data()

    training_data_X, training_data_y = Processing(
        folder_name='ant').transfrom_data(dataset_ant)

    testing_data_X, testing_data_y = Processing(
        folder_name='camel').transfrom_data(dataset_camel)
    print('raw train shape :', training_data_X.shape)
    print('test shape :', testing_data_X.shape)
    nn = NN_filter(training_data_X, training_data_X,
                   testing_data_X, testing_data_y)

    training_nn_X, training_nn_y, testing_nn_X, testing_nn_y = nn.transform()
    print('nn train shape :', training_nn_X.shape)
