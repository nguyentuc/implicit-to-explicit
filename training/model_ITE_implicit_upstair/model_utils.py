import heapq
import logging
import math
import os

import numpy as np
import progressbar
import tensorflow as tf
from terminaltables import AsciiTable

from src.config import config
from src.data_preparation import data_utils
from src.training.model_ITE_implicit_upstair import model_graph

# config log
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class Utils:
    def __init__(self, hyper_params,
                 save_eval_result=True,
                 save_model=False):
        """
        This class contains functions for training
        :type save_model: bool
        :type save_eval_result: bool
        :type hyper_params: dict
        """

        self.root_path = config.DATA_ROOT_PATH + "site_data/" + hyper_params["data_name"]

        self.eval_result_path = ""
        # self.root_path + "eval_results/{}/{}/{}_{}_{}_{}_{}"
        # model_name/epoch/{num_factor}_{eta}_{batch_size}_{lr}_{num_neg}

        self.model_path = ""
        # self.root_path + "models/{}/{}/{}_{}_{}_{}_{}"
        # model_name/epoch/{num_factor}_{eta}_{batch_size}_{lr}_{num_neg}

        self.hyper_params = hyper_params

        self.save_eval_result = save_eval_result  # boolean
        self.save_model = save_model  # boolean

        self.result_str = ""

        self.data = self.load_data()

    @staticmethod
    def show_result_in_key_value(tuple_data):
        table_data = [["key", "values"]]
        for i in tuple_data:
            table_data.append([i[0], i[1]])
        table_data_str = AsciiTable(table_data).table
        print(table_data_str)
        return str(table_data_str)

    def load_data(self):
        """
        Load train, validation, test data and other data
        """
        logging.info("--------------> JOB INFO: " + self.hyper_params["model_name"])

        logging.info("--------------> Loading data ...")
        num_user, num_item = data_utils.load_representation_data(
            self.root_path + "u2index.txt",
            self.root_path + "i2index.txt")
        self.hyper_params["num_user"] = num_user
        self.hyper_params["num_item"] = num_item
        interact_mat = data_utils.load_interact_matrix(self.root_path + "scene_1/_explicit.train.rating", num_user,
                                                       num_item)
        # training_dict = data_utils.load_train_data(self.train_path + "scene_1/_explicit.train.rating")
        test_data = data_utils.load_test_data(self.root_path + "scene_1/_explicit.test.rating")
        negative_data = data_utils.load_negative_data(self.root_path + "scene_1/_explicit.test.negative")

        return {
            # "training_dict": training_dict,
            "interact_mat": interact_mat,
            "test_data": test_data,
            "negative_data": negative_data
        }
        pass

    def set_hyper_params(self, hyper_params, save_eval_result, save_model):
        """
        Load hyper_parameters: batch_size, eta, num_factor, num_negative, lambda, ...
        """
        self.hyper_params["num_factor"] = hyper_params["num_factor"]
        self.hyper_params["eta"] = hyper_params["eta"]
        self.hyper_params["batch_size"] = hyper_params["batch_size"]
        self.hyper_params["lr"] = hyper_params["lr"]
        self.hyper_params["num_neg"] = hyper_params["num_neg"]
        self.hyper_params["num_epoch"] = hyper_params["num_epoch"]
        self.eval_result_path = self.root_path + "eval_results/{}/{}/{}_{}_{}_{}_{}".format(hyper_params["model_name"],
                                                                                            hyper_params["num_epoch"],
                                                                                            hyper_params["num_factor"],
                                                                                            hyper_params["eta"],
                                                                                            hyper_params["batch_size"],
                                                                                            hyper_params["lr"],
                                                                                            hyper_params["num_neg"])
        eval_parent_dir_path = "/".join(self.eval_result_path.split("/")[:-1])
        if not os.path.isdir(eval_parent_dir_path):
            os.makedirs(eval_parent_dir_path)

        self.model_path = self.root_path + "models/{}/{}/{}_{}_{}_{}_{}".format(hyper_params["model_name"],
                                                                                hyper_params["num_epoch"],
                                                                                hyper_params["num_factor"],
                                                                                hyper_params["eta"],
                                                                                hyper_params["batch_size"],
                                                                                hyper_params["lr"],
                                                                                hyper_params["num_neg"])
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.save_eval_result = save_eval_result  # boolean
        self.save_model = save_model  # boolean

        self.result_str = ""

    def build_graph(self):
        """
        Build the model graph with tensorflow
        """
        return model_graph.ITE.create_model(self.hyper_params)

    def evaluate(self, model, top_k, test_data, negative_data, prediction):
        """
        Compute the average evaluation value for all users
        """
        hits, ndcgs = [], []
        # Single thread
        widgets = [progressbar.Percentage(), " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]
        for idx in progressbar.ProgressBar(widgets=widgets)(range(len(test_data))):
            hr, ndcg = self.eval_one_rating(model, idx, top_k, test_data, negative_data, prediction)
            hits.append(hr)
            ndcgs.append(ndcg)
        return np.array(hits).mean(), np.array(ndcgs).mean()

    def eval_one_rating(self, model, idx, top_k, test_data, negative_data, prediction):
        """
        Compute the evaluation value for one user
        """
        rating = test_data[idx]
        user = rating[0]
        gt_item = rating[1]
        items = negative_data[user]
        items.append(gt_item)
        # Get prediction scores
        map_item_score = {}
        # users = np.full(len(items), u, dtype="int64")
        users = [user] * len(items)
        predictions = self.predict(model, users, items, prediction)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()
        # Evaluate top rank list
        rank_list = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)

        hr = self.get_hit_ratio(rank_list, gt_item)
        ndcg = self.get_ndcg(rank_list, gt_item)
        return hr, ndcg

    def predict(self, model, users, items, prediction):
        """
        Get the output prediction scores
        """
        return model["sess"].run(prediction,
                                 feed_dict={model["user_indices_ph"]: users,
                                            model["item_indices_ph"]: items})

    @staticmethod
    def get_hit_ratio(rank_list, gt_item):
        for item in rank_list:
            if item == gt_item:
                return 1.0
        return 0

    @staticmethod
    def get_ndcg(rank_list, gt_item):
        for i in range(len(rank_list)):
            item = rank_list[i]
            if item == gt_item:
                return math.log(2) / math.log(i + 2)
        return 0

    def train(self, model, data):
        # hyper_parameters
        num_epoch = self.hyper_params["num_epoch"]
        num_neg = self.hyper_params["num_neg"]
        batch_size = self.hyper_params["batch_size"]
        verbose = self.hyper_params["verbose"]
        eval_top_k = self.hyper_params["eval_top_k"]
        num_user = self.hyper_params["num_user"]
        num_item = self.hyper_params["num_item"]

        # data
        interact_mat = data["interact_mat"]
        test_data = data["test_data"]
        negative_data = data["negative_data"]

        # model function and placeholder
        user_indices_ph = model["user_indices_ph"]
        item_indices_ph = model["item_indices_ph"]
        labels_ph = model["labels_ph"]
        y1_indicators_ph = model["y1_indicators_ph"]
        y2_indicators_ph = model["y2_indicators_ph"]
        optimizer = model["optimizer"]
        train_loss = model["train_loss"]
        test_loss = model["test_loss"]
        prediction = model["prediction"]

        # ---------------------------------> config gpu for tensorflow <----------------------------------------
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.visible_device_list = self.hyper_params["visible_gpu"]

        # ------------------------------------------> training <------------------------------------------------
        saver = tf.train.Saver()
        with tf.Session(config=session_conf) as sess:
            # init hyper_params (model weights)
            sess.run(tf.global_variables_initializer())
            model["sess"] = sess

            # init evaluation values
            explicit_best_hit, explicit_best_ndcg = [0, 0], [0, 0]

            print("Evaluating ........")
            explicit_hit, explicit_ndcg = self.evaluate(model,
                                                        eval_top_k,
                                                        test_data,
                                                        negative_data,
                                                        prediction)

            if explicit_ndcg > explicit_best_ndcg[0]:
                explicit_best_hit = [explicit_hit, "init"]
                explicit_best_ndcg = [explicit_ndcg, "init"]

            # log init eval_result
            self.result_str += "JOB INFO: " + self.hyper_params["model_name"] + "\n\n" + Utils.show_result_in_key_value(
                self.hyper_params.items()) + "\n\n"
            eval_result_data = [["Epoch", "Train_loss", "Test_loss", "Hit", "NDCG"],
                                ["init", "_", "_", explicit_hit, explicit_ndcg]]
            print(AsciiTable(eval_result_data).table)

            # train model through epochs
            for e in range(1, num_epoch + 1):
                logging.info("-----------> Epoch: " + str(e))
                r_train_loss = 0.0
                r_test_loss = 0.0
                num_batch = 0
                partitioned_train_path = self.root_path + "scene_1/partitioned_train_data/"

                for partition_name in sorted(os.listdir(partitioned_train_path)):
                    print(str(e) + ":" + partition_name)
                    partitioned_path = partitioned_train_path + partition_name
                    # -----------------------> get train instances <-------------------------------
                    user_indices, item_indices, labels, y1_indicators, y2_indicators = \
                        data_utils.get_train_instances_partition(partitioned_path, interact_mat, num_neg, num_item)

                    widgets = [" ", " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]
                    for b in progressbar.ProgressBar(widgets=widgets)(range(0, len(user_indices), batch_size)):
                        batch_u = user_indices[b: b + batch_size]
                        batch_i = item_indices[b: b + batch_size]
                        batch_label = labels[b: b + batch_size]
                        batch_y1_indicator = y1_indicators[b: b + batch_size]
                        batch_y2_indicator = y2_indicators[b: b + batch_size]

                        sess.run(optimizer,
                                 feed_dict={
                                     user_indices_ph: batch_u,
                                     item_indices_ph: batch_i,
                                     labels_ph: batch_label,
                                     y1_indicators_ph: batch_y1_indicator,
                                     y2_indicators_ph: batch_y2_indicator
                                 })
                        if e == 1 or e % verbose == 0:
                            r_loss_tmp = sess.run(test_loss,
                                                  feed_dict={user_indices_ph: batch_u,
                                                             item_indices_ph: batch_i,
                                                             labels_ph: batch_label,
                                                             y1_indicators_ph: batch_y1_indicator,
                                                             y2_indicators_ph: batch_y2_indicator})
                            r_train_loss_tmp = sess.run(train_loss,
                                                        feed_dict={user_indices_ph: batch_u,
                                                                   item_indices_ph: batch_i,
                                                                   labels_ph: batch_label,
                                                                   y1_indicators_ph: batch_y1_indicator,
                                                                   y2_indicators_ph: batch_y2_indicator})
                            r_test_loss += r_loss_tmp
                            r_train_loss += r_train_loss_tmp
                        num_batch += 1
                    # end for each batch
                # end for each data partition
                # Compute loss need to divide into batches, to avoid out of memory error
                r_test_loss /= num_batch
                r_train_loss /= num_batch

                # ------------------------> Evaluate <---------------------------
                if e == 1 or e % verbose == 0:
                    # log for explicit
                    raw_explicit_top = self.predict(model, user_indices, item_indices, prediction)
                    dict_explicit_top = {i: raw_explicit_top[i] for i in range(len(raw_explicit_top))}
                    explicit_top = {i: dict_explicit_top[i] for i in
                                    heapq.nlargest(6, dict_explicit_top, key=dict_explicit_top.get)}
                    print(explicit_top)

                    print("Evaluating ........")
                    explicit_hit, explicit_ndcg = self.evaluate(model,
                                                                eval_top_k,
                                                                test_data,
                                                                negative_data,
                                                                prediction)

                    if explicit_ndcg > explicit_best_ndcg[0]:
                        explicit_best_hit = [explicit_hit, e]
                        explicit_best_ndcg = [explicit_ndcg, e]

                    # log eval_result
                    eval_result_data.append([str(e), r_train_loss, r_test_loss, explicit_hit, explicit_ndcg])
                    print(AsciiTable(eval_result_data).table)
                    if self.save_eval_result:
                        with open(self.eval_result_path, "w") as log:
                            log.write(self.result_str + AsciiTable(eval_result_data).table)
                # end evaluate

                if self.save_model:
                    saver.save(sess, self.model_path + "/model")
            # end for each epoch

            logging.info("---------------------------> DONE TRAINING ==> RESULT <------------------------------------")
            best = {"explicit_best_hit": explicit_best_hit,
                    "explicit_best_ndcg": explicit_best_ndcg}
            Utils.show_result_in_key_value(self.hyper_params.items())

            if self.save_eval_result:
                with open(self.eval_result_path, "w") as log:
                    log.write(self.result_str + AsciiTable(eval_result_data).table
                              + "\n\n" + Utils.show_result_in_key_value(best.items()))

    def run(self, hyper_params, save_eval_result, save_model):
        """
        Load hyper_parameters, re-init model params and train
        :type save_model: bool
        :type save_eval_result: bool
        :type hyper_params: dict
        """
        # data = self.load_data()
        self.set_hyper_params(hyper_params, save_eval_result, save_model)
        model = self.build_graph()
        self.train(model, self.data)
