# lib
import csv
import logging
import random
import numpy as np
import progressbar
import scipy.sparse as sp
import datetime

from src.config import config
from src.data_preparation.sparse_vector import *
import scipy


def load_representation_data(u2index_path, i2index_path):
    u2index = {}
    i2index = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    return num_user, num_item


def load_representation_data_with_item_repr(u2index_path, i2index_path, item_repr_path):
    u2index = {}
    i2index = {}
    item_repr = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    count = 0

    with open(item_repr_path) as f:
        item_pcat_dimension = int(next(f))
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            item_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])
    print('Dimension of item pcat: ', item_pcat_dimension)
    print('Num item (check again): ', len(item_repr))

    return num_user, num_item, item_repr, item_pcat_dimension


def load_representation_data_with_both_user_item_repr(u2index_path, i2index_path, user_repr_path, item_repr_path):
    u2index = {}
    i2index = {}
    user_repr = {}
    item_repr = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    count = 0

    with open(user_repr_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            user_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])
    print('Num user have pcat: ', len(user_repr))
    count = 0

    with open(item_repr_path) as f:
        item_pcat_dimension = int(next(f))
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            item_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])

    print('Dimension of item pcat: ', item_pcat_dimension)
    print('Num item (check again): ', len(item_repr))

    return num_user, num_item, user_repr, item_repr, item_pcat_dimension


def load_interact_matrix(file_path, num_user, num_item):
    start = datetime.datetime.now()
    # Construct matrix
    mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            uid, itemid = int(line[0]), int(line[1])
            mat[uid, itemid] = True
    print("time load_interact_matrix: ", datetime.datetime.now() - start)
    return mat


def load_interact_matrix_NeuMF(file_path, num_user, num_item, use_implicit):
    start = datetime.datetime.now()
    # Construct matrix
    mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    if use_implicit:
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            for line in progressbar.ProgressBar()(csv_reader):
                uid, itemid = int(line[0]), int(line[1])
                mat[uid, itemid] = True
    else:
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            for line in progressbar.ProgressBar()(csv_reader):
                uid, itemid, explicit_timestamp = int(line[0]), int(line[1]), int(line[4])
                if explicit_timestamp > 0:
                    mat[uid, itemid] = True
    print("time load_interact_matrix: ", datetime.datetime.now() - start)
    return mat


'''
def load_train_data(file_path):
    # Construct user rating dict
    user_im_dict = {}
    user_ex_dict = {}
    training_dict = {}
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            uid, itemid, explicit_timestamp = int(line[0]), int(line[1]), int(line[4])
            # if explicit_timestamp > 0:
            #     if uid not in user_ex_dict:
            #         user_ex_dict[uid] = [itemid]
            #     else:
            #         user_ex_dict[uid].append(itemid)
            # else:
            #     if uid not in user_im_dict:
            #         user_im_dict[uid] = [itemid]
            #     else:
            #         user_im_dict[uid].append(itemid)
            if explicit_timestamp > 0:
                training_dict[(uid, itemid)] = True
            else:
                training_dict[(uid, itemid)] = False
    return training_dict
'''


def load_test_data(file_path):
    test_list = []
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            user, item = int(line[0]), int(line[1])
            test_list.append([user, item])
    print('len test data: ', len(test_list))
    return test_list


# def load_negative_data(file_path):
#     negative_dict = {}
#     with open(file_path, "r") as f:
#         csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
#         # i = 0
#         for line in progressbar.ProgressBar()(csv_reader):
#             user = line[0].split(",")
#             user = int(user[0][1:])
#             # assert user == i
#             # i += 1
#
#             negative_dict[user] = []
#             for x in line[1:]:
#                 negative_dict[user].append(int(x))
#     print('len negative data: ', len(negative_dict))
#     return negative_dict


def load_negative_data(file_path):
    negative_dict = {}
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        # i = 0
        for line in progressbar.ProgressBar()(csv_reader):
            user = int(line[0])
            # assert user == i
            # i += 1

            negative_dict[user] = []
            for x in line[2:]:
                negative_dict[user].append(int(x))
    print('len negative data: ', len(negative_dict))
    return negative_dict


def get_train_instances(training_dict, num_negatives, num_user, num_item):
    user_input, item_input, labels, indicator = [], [], [], []
    start = datetime.datetime.now()
    set_total_items = set(range(num_item))
    for (uid, itemid) in progressbar.ProgressBar()(training_dict):
        is_ex = training_dict[(uid, itemid)]
        user_input.append(uid)
        item_input.append(itemid)
        labels.append(1)
        # im_indicator.append(1)
        if is_ex:
            indicator.append(1)
        else:
            indicator.append(0)

        # negative instances
        # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
        for _ in range(num_negatives):
            j = random.randrange(num_item)
            while (uid, j) in training_dict:
                j = random.randrange(num_item)
            # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
            user_input.append(uid)
            item_input.append(j)
            labels.append(0)
            indicator.append(0)
        # end for

    print("time get_train_instances: ", datetime.datetime.now() - start)
    return user_input, item_input, labels, indicator


def get_train_instances_partition(file_path, interact_mat, num_negatives, num_item):
    user_indices, item_indices, labels, y1_indicators, y2_indicators = [], [], [], [], []
    start = datetime.datetime.now()
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            user_id, item_id, interact_type = int(line[0]), int(line[1]), int(line[3])

            is_ex = interact_type > 1
            if not is_ex:
                user_indices.append(user_id)
                item_indices.append(item_id)
                labels.append(1)
                # im_indicator.append(1)
                y1_indicators.append(1)
                y2_indicators.append(0)

                # negative instances
                # Lay mau cac item chua tung view (negative instances for implicit record)

                for _ in range(num_negatives):
                    j = random.randrange(num_item)
                    while (user_id, j) in interact_mat:
                        j = random.randrange(num_item)
                    # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                    user_indices.append(user_id)
                    item_indices.append(j)
                    labels.append(0)
                    y1_indicators.append(1)
                    y2_indicators.append(0)
            else:
                user_indices.append(user_id)
                item_indices.append(item_id)
                labels.append(1)
                # im_indicator.append(1)
                y1_indicators.append(0)
                y2_indicators.append(1)

                # negative instances
                # Lay mau cac item chua tung co tuong tac explicit (negative instances for explicit record)
                for _ in range(num_negatives):
                    j = random.randrange(num_item)
                    while (user_id, j) in interact_mat:
                        j = random.randrange(num_item)
                    # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                    user_indices.append(user_id)
                    item_indices.append(j)
                    labels.append(0)
                    y1_indicators.append(0)
                    y2_indicators.append(1)
            # end for
    print("time get_train_instances: {} s".format((datetime.datetime.now() - start).total_seconds()))
    return user_indices, item_indices, labels, y1_indicators, y2_indicators


def get_train_instances_partition_NeuMF(file_path, interact_mat, num_negatives, num_item, use_implicit):
    user_indices, item_indices, labels = [], [], []
    start = datetime.datetime.now()
    if use_implicit:
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            for line in progressbar.ProgressBar()(csv_reader):
                uid, itemid = int(line[0]), int(line[1])
                user_indices.append(uid)
                item_indices.append(itemid)
                labels.append(1)
                # negative instances
                # Lay mau cac item chua tung view
                for _ in range(num_negatives):
                    j = random.randrange(num_item)
                    while (uid, j) in interact_mat:
                        j = random.randrange(num_item)
                    # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                    user_indices.append(uid)
                    item_indices.append(j)
                    labels.append(0)
                # end for each negative sample
            # end for each line
    else:
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            for line in progressbar.ProgressBar()(csv_reader):
                uid, itemid, explicit_timestamp = int(line[0]), int(line[1]), int(line[4])
                if explicit_timestamp <= 0:
                    continue
                user_indices.append(uid)
                item_indices.append(itemid)
                labels.append(1)
                # negative instances
                # Lay mau cac item chua tung view
                for _ in range(num_negatives):
                    j = random.randrange(num_item)
                    while (uid, j) in interact_mat:
                        j = random.randrange(num_item)
                    # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                    user_indices.append(uid)
                    item_indices.append(j)
                    labels.append(0)
                # end for each negative sample
            # end for each line
    print("time get_train_instances: ", datetime.datetime.now() - start)
    return user_indices, item_indices, labels


# def get_train_instances(train_data_mt, num_negatives, num_items):
#     user_input, item_input, implicit_labels, explicit_labels = [], [], [], []
#     start = datetime.datetime.now()
#
#     for (u, i) in train_data_mt.keys():
#         # positive instance
#         user_input.append(u)
#         item_input.append(i)
#         implicit_labels.append(1)
#         if train_data_mt[u, i] > 0:
#             explicit_labels.append(1)
#         else:
#             explicit_labels.append(0)
#
#         # negative instances
#         # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
#         for _ in range(num_negatives):
#             j = random.randrange(num_items)
#             while (u, j) in train_data_mt:
#                 j = random.randrange(num_items)
#             # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
#             user_input.append(u)
#             item_input.append(j)
#             implicit_labels.append(0)
#             explicit_labels.append(0)
#         # end for
#
#     print("time get_train_instances: ", datetime.datetime.now() - start)
#     return user_input, item_input, implicit_labels, explicit_labels


def count_im_ex(file_path):
    start = datetime.datetime.now()
    # Construct matrix
    # mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    count_ex = 0
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            ex = int(line[5])
            if ex > 0:
                count_ex += 1
    print(count_ex)


if __name__ == '__main__':
    retail_rocket_root_path = config.DATA_ROOT_PATH + 'site_data/retail_rocket/'
    recobell_root_path = config.DATA_ROOT_PATH + "site_data/recobell/"
    ml_1m_root_path = config.DATA_ROOT_PATH + "site_data/ml-1m/"
    count_im_ex(retail_rocket_root_path + "scene_1/_explicit.train.rating")
    count_im_ex(recobell_root_path + "scene_1/_explicit.train.rating")
