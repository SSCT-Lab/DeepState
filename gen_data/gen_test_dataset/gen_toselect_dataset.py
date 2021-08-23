import argparse
import os
import numpy as np
import pandas as pd
from keras.datasets import mnist
import random


def gen_mnist():
    x_test_data_path = "./dau/mnist_harder/x_test_0.npy"
    y_test_data_path = "./dau/mnist_harder/y_test_0.npy"
    ori_x_test_data_path = "./dau/mnist_harder/x_ori_test_0.npy"
    ori_y_test_data_path = "./dau/mnist_harder/y_ori_test_0.npy"

    x_test_data = np.load(x_test_data_path)
    y_test_data = np.load(y_test_data_path)
    ori_x_test_data = np.load(ori_x_test_data_path)
    ori_y_test_data = np.load(ori_y_test_data_path)

    # print(x_test_data.shape)  # 10000
    os.makedirs("../../gen_data/mnist_toselect", exist_ok=True)

    li = np.arange(len(x_test_data))
    for times in range(30):  # 30 different samples
        x_to_select_mnist = []
        y_to_select_mnist = []
        select_id = np.random.choice(a=li, size=3000, replace=False)
        print(select_id)

        for i in select_id:
            d1 = x_test_data[i].reshape(1, 28, 28)
            d2 = ori_x_test_data[i].reshape(1, 28, 28)
            x_to_select_mnist.append(d1)
            y_to_select_mnist.append(y_test_data[i])
            x_to_select_mnist.append(d2)
            y_to_select_mnist.append(ori_y_test_data[i])

        x_save_data = np.array(x_to_select_mnist)
        y_save_data = np.array(y_to_select_mnist)
        state = np.random.get_state()
        np.random.shuffle(x_save_data)
        np.random.set_state(state)
        np.random.shuffle(y_save_data)

        np.savez(("../../gen_data/mnist_toselect/mnist_toselect" + "_{}").format(times), X=x_save_data, Y=y_save_data)


def gen_snips():
    data = pd.read_csv("./dau/snips_harder/snips_toselect.csv")
    length = int(len(data) / 2)
    print("ori/aug legth:", length)
    ori = data[:int(len(data) / 2)]
    aug = data[int(len(data) / 2):]
    li = np.arange(len(ori))

    os.makedirs("../../gen_data/snips_toselect", exist_ok=True)

    for times in range(30):
        text, intent = [], []
        to_select = pd.DataFrame(columns=('text', 'intent'))
        select_id = np.random.choice(a=li, size=1000, replace=False)
        print(select_id)

        for i in select_id:
            text.append(data.text[i])
            text.append(data.text[i + length])
            intent.append(data.intent[i])
            intent.append(data.intent[i + length])

        for idx, (text_i, intent_i) in enumerate(zip(text, intent)):
            tmp = {'text': text_i, 'intent': intent_i}
            to_select.loc[idx] = tmp

        to_select.sample(frac=1)  # shuffle
        to_select.to_csv(("../../gen_data/snips_toselect/snips_toselect" + "_{}" + ".csv").format(times))


def gen_fashion():
    x_test_data_path = "./dau/fashion_harder/x_test_0.npy"
    y_test_data_path = "./dau/fashion_harder/y_test_0.npy"
    ori_x_test_data_path = "./dau/fashion_harder/x_ori_test_0.npy"
    ori_y_test_data_path = "./dau/fashion_harder/y_ori_test_0.npy"

    x_test_data = np.load(x_test_data_path)
    y_test_data = np.load(y_test_data_path)
    ori_x_test_data = np.load(ori_x_test_data_path)
    ori_y_test_data = np.load(ori_y_test_data_path)

    # print(x_test_data.shape)  # 10000
    os.makedirs("../../gen_data/fashion_toselect", exist_ok=True)

    li = np.arange(len(x_test_data))
    for times in range(30):
        x_to_select_mnist = []
        y_to_select_mnist = []
        select_id = np.random.choice(a=li, size=3000, replace=False)
        print(select_id)

        for i in select_id:
            d1 = x_test_data[i].reshape(1, 28, 28)
            d2 = ori_x_test_data[i].reshape(1, 28, 28)
            x_to_select_mnist.append(d1)
            y_to_select_mnist.append(y_test_data[i])
            x_to_select_mnist.append(d2)
            y_to_select_mnist.append(ori_y_test_data[i])

        x_save_data = np.array(x_to_select_mnist)
        y_save_data = np.array(y_to_select_mnist)
        state = np.random.get_state()
        np.random.shuffle(x_save_data)
        np.random.set_state(state)
        np.random.shuffle(y_save_data)

        np.savez(("../../gen_data/fashion_toselect/fashion_toselect" + "_{}").format(times), X=x_save_data, Y=y_save_data)


def gen_agnews():
    data = pd.read_csv("./dau/agnews_harder/agnews_toselect.csv")
    length = int(len(data) / 2)
    print("ori/aug legth:", length)
    ori = data[:int(len(data) / 2)]
    aug = data[int(len(data) / 2):]
    li = np.arange(len(ori))

    os.makedirs("../../gen_data/agnews_toselect", exist_ok=True)

    for times in range(30):
        text, intent = [], []
        to_select = pd.DataFrame(columns=('news', 'label'))
        select_id = np.random.choice(a=li, size=2280, replace=False)
        print(select_id)

        for i in select_id:
            text.append(data.news[i])
            text.append(data.news[i + length])
            intent.append(data.label[i])
            intent.append(data.label[i + length])

        for idx, (text_i, intent_i) in enumerate(zip(text, intent)):
            tmp = {'news': text_i, 'label': intent_i}
            to_select.loc[idx] = tmp

        to_select.sample(frac=1)  # shuffle
        to_select.to_csv(("../../gen_data/agnews_toselect/agnews_toselect" + "_{}" + ".csv").format(times))


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generate the dataset for selection.")
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews'])
    args = parse.parse_args()

    if args.dataset == "mnist":
        gen_mnist()
    
    if args.dataset == "snips":
        gen_snips()

    if args.dataset == "fashion":
        gen_fashion()

    if args.dataset == "agnews":
        gen_agnews()
