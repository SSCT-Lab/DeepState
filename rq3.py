import argparse
from statics import *
import numpy as np
import pandas as pd
import os
from selection_tools import get_selection_information, get_selected_data, get_val_data
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate the inclusiveness for the selected dataset.")
    parse.add_argument('-dl_model', help='path of dl model', required=True)
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion'])
    args = parse.parse_args()

    if args.model_type == "lstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier

        lstm_classifier = MnistLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/mnist_retrain/mnist_toselect.npz"
        ori_val_path = "./gen_data/mnist_retrain/mnist_ori_test.npz"
        aug_val_path = "./gen_data/mnist_retrain/mnist_aug_test.npz"
        mix_val_path = "./gen_data/mnist_retrain/mnist_mix_test.npz"
        retrain_save_path = "./RNNModels/mnist_demo/models/lstm_selected_"
        wrapper_path = "./RNNModels/mnist_demo/output/lstm/abst_model/wrapper_lstm_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier

        lstm_classifier = MnistBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = MnistBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/mnist_retrain/mnist_toselect.npz"
        ori_val_path = "./gen_data/mnist_retrain/mnist_ori_test.npz"
        aug_val_path = "./gen_data/mnist_retrain/mnist_aug_test.npz"
        mix_val_path = "./gen_data/mnist_retrain/mnist_mix_test.npz"
        retrain_save_path = "./RNNModels/mnist_demo/models/blstm_selected_"
        wrapper_path = "./RNNModels/mnist_demo/output/blstm/abst_model/wrapper_blstm_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier

        lstm_classifier = SnipsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/snips_retrain/snips_toselect.csv"
        ori_val_path = "./gen_data/snips_retrain/snips_ori_test.csv"
        aug_val_path = "./gen_data/snips_retrain/snips_aug_test.csv"
        mix_val_path = "./gen_data/snips_retrain/snips_mix_test.csv"
        retrain_save_path = "./RNNModels/snips_demo/models/blstm_selected_"
        wrapper_path = "./RNNModels/snips_demo/output/blstm/abst_model/wrapper_blstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "gru" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier

        lstm_classifier = SnipsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = SnipsGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/snips_retrain/snips_toselect.csv"
        ori_val_path = "./gen_data/snips_retrain/snips_ori_test.csv"
        aug_val_path = "./gen_data/snips_retrain/snips_aug_test.csv"
        mix_val_path = "./gen_data/snips_retrain/snips_mix_test.csv"
        retrain_save_path = "./RNNModels/snips_demo/models/gru_selected_"
        wrapper_path = "./RNNModels/snips_demo/output/gru/abst_model/wrapper_gru_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "lstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier

        lstm_classifier = FashionLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionLSTMClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/fashion_retrain/fashion_toselect.npz"
        ori_val_path = "./gen_data/fashion_retrain/fashion_ori_test.csv"
        aug_val_path = "./gen_data/fashion_retrain/fashion_aug_test.csv"
        mix_val_path = "./gen_data/fashion_retrain/fashion_mix_test.csv"
        retrain_save_path = "./RNNModels/fashion_demo/models/lstm_selected_"
        wrapper_path = "./RNNModels/fashion_demo/output/lstm/abst_model/wrapper_lstm_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "gru" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier

        lstm_classifier = FashionGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(args.dl_model)
        dense_classifier = FashionGRUClassifier()
        dense_model = dense_classifier.reload_dense(args.dl_model)

        to_select_path = "./gen_data/fashion_retrain/fashion_toselect.npz"
        ori_val_path = "./gen_data/fashion_retrain/fashion_ori_test.csv"
        aug_val_path = "./gen_data/fashion_retrain/fashion_aug_test.csv"
        mix_val_path = "./gen_data/fashion_retrain/fashion_mix_test.csv"
        retrain_save_path = "./RNNModels/fashion_demo/models/gru_selected_"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 6000

    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    ori_acc_imp, aug_acc_imp, mix_acc_imp = {}, {}, {}

    pre_li = [1, 5, 10, 15, 20]
    weight_state, unique_index_arr_id, stellar_bscov, stellar_btcov, rnntest_sc, nc_cov, nc_cam, \
    rnntest_sc_cam, trend_set, right = get_selection_information(to_select_path, model, lstm_classifier,
                                                                 dense_model, wrapper_path, w2v_path, time_steps)

    select_method = ['state_w_selected', 'random_selected', 'cov_selected', 'bscov_selected', 'btcov_selected',
                     'sc_ctm_selected', 'sc_cam_selected', 'nc_ctm_selected', 'nc_cam_selected']
    for item in select_method:
        ori_acc_imp[item] = []
        aug_acc_imp[item] = []
        mix_acc_imp[item] = []

    for pre in pre_li:
        select_num = int(total_num * 0.01 * pre)

        # selection
        state_w_selected = selection(weight_state, trend_set, select_num)
        random_selected = ran_selection(total_num, select_num)
        cov_selected = cam_selection(unique_index_arr_id, total_num, select_num)
        bscov_selected = ctm_selection(np.array(stellar_bscov), total_num, select_num)
        btcov_selected = ctm_selection(np.array(stellar_btcov), total_num, select_num)
        sc_ctm_selected = ctm_selection(np.array(rnntest_sc), total_num, select_num)
        sc_cam_selected = nc_cam_selection(np.array(rnntest_sc_cam), total_num, select_num)
        nc_ctm_selected = ctm_selection(np.array(nc_cov), total_num, select_num)
        nc_cam_selected = nc_cam_selection(np.array(nc_cam), total_num, select_num)

        x_ori_val, y_ori_val = get_val_data(ori_val_path, w2v_path)
        x_aug_val, y_aug_val = get_val_data(aug_val_path, w2v_path)
        x_mix_val, y_mix_val = get_val_data(mix_val_path, w2v_path)

        for method_item in select_method:
            X_selected_array, Y_selected_array = get_selected_data(to_select_path, np.array(eval(method_item)), w2v_path)
            retrained_model_path = retrain_save_path + str(pre) + "/" + str(method_item) + "_" + \
                                   str(args.dataset) + "_" + str(args.model_type) + ".h5"
            if os.path.isfile(retrained_model_path):
                print("Have already saved the retrained model.")
                break
            os.makedirs(retrain_save_path + str(pre), exist_ok=True)
            lstm_classifier.retrain(X_selected_array, Y_selected_array, x_ori_val, y_ori_val, retrained_model_path)

            K.clear_session()
            ori_acc_imp[method_item].append(lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_ori_val, y_ori_val))
            aug_acc_imp[method_item].append(lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_aug_val, y_aug_val))
            mix_acc_imp[method_item].append(lstm_classifier.evaluate_retrain(retrained_model_path, args.dl_model, x_mix_val, y_mix_val))

    result_dict = {}
    result_dict['select rate'] = pre_li
    for method_item in select_method:
        result_dict[str(method_item) + str("_ori_acc_imp")] = ori_acc_imp[method_item]
        result_dict[str(method_item) + str("_aug_acc_imp")] = aug_acc_imp[method_item]
        result_dict[str(method_item) + str("_mix_acc_imp")] = mix_acc_imp[method_item]

    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/rq3", exist_ok=True)
    df.to_csv("./exp_results/rq3/rq3_{}_{}.csv".format(args.dataset, args.model_type))