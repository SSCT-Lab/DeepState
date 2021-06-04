import argparse
import os
import numpy as np
import pandas as pd


def gen_mnist():
    x_test_data_path = "./gen_data/dau/mnist_harder/x_test_0.npy"
    y_test_data_path = "./gen_data/dau/mnist_harder/y_test_0.npy"
    ori_x_test_data_path = "./gen_data/dau/mnist_harder/x_ori_test_0.npy"
    ori_y_test_data_path = "./gen_data/dau/mnist_harder/y_ori_test_0.npy"

    x_test_data = np.load(x_test_data_path)
    y_test_data = np.load(y_test_data_path)
    ori_x_test_data = np.load(ori_x_test_data_path)
    ori_y_test_data = np.load(ori_y_test_data_path)

    li = np.arange(len(x_test_data))
    x_to_select_mnist, y_to_select_mnist = [], []
    select_id = np.random.choice(a=li, size=3000, replace=False)
    print("selected id:", select_id)
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    retrain_test_id = np.delete(li, select_id)
    print("retrain id:", retrain_test_id)

    for i in select_id:
        d1 = x_test_data[i].reshape(1, 28, 28)
        d2 = ori_x_test_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(y_test_data[i])
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(ori_y_test_data[i])
    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)

    for j in retrain_test_id:
        dd1 = x_test_data[j].reshape(1, 28, 28)
        dd2 = ori_x_test_data[j].reshape(1, 28, 28)
        x_retrain_aug_test.append(dd1)
        x_retrain_ori_test.append(dd2)
        y_retrain_aug_test.append(y_test_data[j])
        y_retrain_ori_test.append(ori_y_test_data[j])

        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(y_test_data[j])
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(ori_y_test_data[j])
    x_ori_test = np.array(x_retrain_ori_test)
    y_ori_test = np.array(y_retrain_ori_test)
    x_aug_test = np.array(x_retrain_aug_test)
    y_aug_test = np.array(y_retrain_aug_test)
    x_mix_test = np.array(x_retrain_mix_test)
    y_mix_test = np.array(y_retrain_mix_test)

    os.makedirs("./gen_data/mnist_retrain", exist_ok=True)

    np.savez("./gen_data/mnist_retrain/mnist_toselect", X=x_save_data, Y=y_save_data)
    np.savez("./gen_data/mnist_retrain/mnist_ori_test", X=x_ori_test, Y=y_ori_test)
    np.savez("./gen_data/mnist_retrain/mnist_aug_test", X=x_aug_test, Y=y_aug_test)
    np.savez("./gen_data/mnist_retrain/mnist_mix_test", X=x_mix_test, Y=y_mix_test)


def gen_fashion():
    x_test_data_path = "./gen_data/dau/fashion_harder/x_test_0.npy"
    y_test_data_path = "./gen_data/dau/fashion_harder/y_test_0.npy"
    ori_x_test_data_path = "./gen_data/dau/fashion_harder/x_ori_test_0.npy"
    ori_y_test_data_path = "./gen_data/dau/fashion_harder/y_ori_test_0.npy"

    x_test_data = np.load(x_test_data_path)
    y_test_data = np.load(y_test_data_path)
    ori_x_test_data = np.load(ori_x_test_data_path)
    ori_y_test_data = np.load(ori_y_test_data_path)

    li = np.arange(len(x_test_data))
    x_to_select_mnist, y_to_select_mnist = [], []
    select_id = np.random.choice(a=li, size=3000, replace=False)
    print("selected id:", select_id)
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    retrain_test_id = np.delete(li, select_id)
    print("retrain id:", retrain_test_id)

    for i in select_id:
        d1 = x_test_data[i].reshape(1, 28, 28)
        d2 = ori_x_test_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(y_test_data[i])
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(ori_y_test_data[i])
    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)

    for j in retrain_test_id:
        dd1 = x_test_data[j].reshape(1, 28, 28)
        dd2 = ori_x_test_data[j].reshape(1, 28, 28)
        x_retrain_aug_test.append(dd1)
        x_retrain_ori_test.append(dd2)
        y_retrain_aug_test.append(y_test_data[j])
        y_retrain_ori_test.append(ori_y_test_data[j])

        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(y_test_data[j])
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(ori_y_test_data[j])
    x_ori_test = np.array(x_retrain_ori_test)
    y_ori_test = np.array(y_retrain_ori_test)
    x_aug_test = np.array(x_retrain_aug_test)
    y_aug_test = np.array(y_retrain_aug_test)
    x_mix_test = np.array(x_retrain_mix_test)
    y_mix_test = np.array(y_retrain_mix_test)

    os.makedirs("./gen_data/fashion_retrain", exist_ok=True)

    np.savez("./gen_data/fashion_retrain/fashion_toselect", X=x_save_data, Y=y_save_data)
    np.savez("./gen_data/fashion_retrain/fashion_ori_test", X=x_ori_test, Y=y_ori_test)
    np.savez("./gen_data/fashion_retrain/fashion_aug_test", X=x_aug_test, Y=y_aug_test)
    np.savez("./gen_data/fashion_retrain/fashion_mix_test", X=x_mix_test, Y=y_mix_test)


def gen_snips():
    data = pd.read_csv("./gen_data/dau/snips_harder/to_select_intent.csv")
    length = int(len(data) / 2)
    print("ori/aug length:", length)
    ori = data[:int(len(data) / 2)]
    li = np.arange(len(ori))

    text, intent = [], []
    text_ori, intent_ori = [], []
    text_aug, intent_aug = [], []
    text_mix, intent_mix = [], []
    to_select = pd.DataFrame(columns=('text', 'intent'))
    ori_test = pd.DataFrame(columns=('text', 'intent'))
    aug_test = pd.DataFrame(columns=('text', 'intent'))
    mix_test = pd.DataFrame(columns=('text', 'intent'))
    select_id = np.random.choice(a=li, size=1000, replace=False)
    print("selected id:", select_id)
    retrain_test_id = np.delete(li, select_id)
    print("retrain id:", retrain_test_id)

    for i in select_id:
        text.append(data.text[i])
        text.append(data.text[i + length])
        intent.append(data.intent[i])
        intent.append(data.intent[i + length])

    for idx, (text_i, intent_i) in enumerate(zip(text, intent)):
        tmp = {'text': text_i, 'intent': intent_i}
        to_select.loc[idx] = tmp

    for j in retrain_test_id:
        text_ori.append(data.text[j])
        text_aug.append(data.text[j + length])
        intent_ori.append(data.intent[j])
        intent_aug.append(data.intent[j + length])

        text_mix.append(data.text[j])
        text_mix.append(data.text[j + length])
        intent_mix.append(data.intent[j])
        intent_mix.append(data.intent[j + length])

    for idx, (text_i, intent_i) in enumerate(zip(text_ori, intent_ori)):
        tmp = {'text': text_i, 'intent': intent_i}
        ori_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_aug, intent_aug)):
        tmp = {'text': text_i, 'intent': intent_i}
        aug_test.loc[idx] = tmp
    for idx, (text_i, intent_i) in enumerate(zip(text_mix, intent_mix)):
        tmp = {'text': text_i, 'intent': intent_i}
        mix_test.loc[idx] = tmp

    os.makedirs("./gen_data/snips_retrain", exist_ok=True)
    to_select.to_csv("./gen_data/snips_retrain/snips_toselect.csv")
    ori_test.to_csv("./gen_data/snips_retrain/snips_ori_test.csv")
    aug_test.to_csv("./gen_data/snips_retrain/snips_aug_test.csv")
    mix_test.to_csv("./gen_data/snips_retrain/snips_mix_test.csv")


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generate the dataset for selection (retrain) and testset (evaluation for the retrained model).")
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion'])
    args = parse.parse_args()

    if args.dataset == "mnist":
        gen_mnist()
    
    if args.dataset == "snips":
        gen_snips()

    if args.dataset == "fashion":
        gen_fashion()
