import argparse
import numpy as np
from statics import *
from selection_tools import get_selection_information
import keras
import datetime
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)


# RQ1: Bug Detection Rate on {10%, 20%, 50%} selected test set.
if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate the bug detection rate for the selected dataset.")
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

        to_select_path = "./gen_data/mnist_toselect"
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

        to_select_path = "./gen_data/mnist_toselect"
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

        to_select_path = "./gen_data/snips_toselect"
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

        to_select_path = "./gen_data/snips_toselect"
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

        to_select_path = "./gen_data/fashion_toselect"
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

        to_select_path = "./gen_data/fashion_toselect"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 6000

    else:
        print("The model and data set are incorrect.")
        sys.exit(1)

    state_w_bdr, ran_bdr, RNNTestcov_bdr, Stellarbscov_bdr, Stellarbtcov_bdr, \
    sc_ctm_bdr, sc_cam_bdr, nc_ctm_bdr, nc_cam_bdr = {}, {}, {}, {}, {}, {}, {}, {}, {}
    pre_li = [10, 20, 50]
    for i in pre_li:
        state_w_bdr[i] = []
        ran_bdr[i] = []
        RNNTestcov_bdr[i] = []
        Stellarbscov_bdr[i] = []
        Stellarbtcov_bdr[i] = []
        sc_ctm_bdr[i] = []
        sc_cam_bdr[i] = []
        nc_ctm_bdr[i] = []
        nc_cam_bdr[i] = []

    files = os.listdir(to_select_path)
    for file in files:
        print("time:", datetime.datetime.now())
        print("Processing file:", file)
        file_path = to_select_path + "/" + file

        weight_state, unique_index_arr_id, stellar_bscov, stellar_btcov, rnntest_sc, nc_cov, nc_cam, \
        rnntest_sc_cam, trend_set, right = get_selection_information(file_path, model, lstm_classifier,
                                                                     dense_model, wrapper_path, w2v_path, time_steps)

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

            state_w_R, state_w_P, _, _, _ = selection_evaluate(right, state_w_selected)
            random_R, random_P, _, _, _ = selection_evaluate(right, random_selected)
            cov_R, cov_P, _, _, _ = selection_evaluate(right, cov_selected)
            bscov_R, bscov_P, _, _, _ = selection_evaluate(right, bscov_selected)
            btcov_R, btcov_P, _, _, _ = selection_evaluate(right, btcov_selected)
            sc_ctm_R, sc_ctm_P, _, _, _ = selection_evaluate(right, sc_ctm_selected)
            sc_cam_R, sc_cam_P, _, _, _ = selection_evaluate(right, sc_cam_selected)
            nc_ctm_R, nc_ctm_P, _, _, _ = selection_evaluate(right, nc_ctm_selected)
            nc_cam_R, nc_cam_P, _, _, _ = selection_evaluate(right, nc_cam_selected)

            state_w_bdr[pre].append(state_w_P)
            ran_bdr[pre].append(random_P)
            RNNTestcov_bdr[pre].append(cov_P)
            Stellarbscov_bdr[pre].append(bscov_P)
            Stellarbtcov_bdr[pre].append(btcov_P)
            sc_ctm_bdr[pre].append(sc_ctm_P)
            sc_cam_bdr[pre].append(sc_cam_P)
            nc_ctm_bdr[pre].append(nc_ctm_P)
            nc_cam_bdr[pre].append(nc_cam_P)

        print(state_w_bdr, ran_bdr, RNNTestcov_bdr, Stellarbscov_bdr, Stellarbtcov_bdr, sc_ctm_bdr)

    result_dict = {'state_w10': state_w_bdr[10], 'state_w20': state_w_bdr[20], 'state_w50': state_w_bdr[50],
                   'random10': ran_bdr[10], 'random20': ran_bdr[20], 'random50': ran_bdr[50],
                   'RNNTestcov10': RNNTestcov_bdr[10], 'RNNTestcov20': RNNTestcov_bdr[20],
                   'RNNTestcov50': RNNTestcov_bdr[50],
                   'Stellarbscov10': Stellarbscov_bdr[10], 'Stellarbscov20': Stellarbscov_bdr[20],
                   'Stellarbscov50': Stellarbscov_bdr[50],
                   'Stellarbtcov10': Stellarbtcov_bdr[10], 'Stellarbtcov20': Stellarbtcov_bdr[20],
                   'Stellarbtcov50': Stellarbtcov_bdr[50],
                   'testRNNsc10': sc_ctm_bdr[10], 'testRNNsc20': sc_ctm_bdr[20], 'testRNNsc50': sc_ctm_bdr[50],
                   'testRNNsc_cam10': sc_cam_bdr[10], 'testRNNsc_cam20': sc_cam_bdr[20], 'testRNNsc_cam50': sc_cam_bdr[50],
                   'nc_ctm10': nc_ctm_bdr[10], 'nc_ctm20': nc_ctm_bdr[20], 'nc_ctm50': nc_ctm_bdr[50],
                   'nc_cam10': nc_cam_bdr[10], 'nc_cam20': nc_cam_bdr[20], 'nc_cam50': nc_cam_bdr[50]}

    print(result_dict)
    df = pd.DataFrame(result_dict)
    os.makedirs("./exp_results/rq1", exist_ok=True)
    df.to_csv("./exp_results/rq1/rq1_{}_{}.csv".format(args.dataset, args.model_type))

    print("Finished! The results are saved in: [./exp_results/rq1/rq1_{}_{}.csv]".format(args.dataset, args.model_type))
