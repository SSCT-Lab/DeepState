import os
import numpy as np


def gen_fashion():
    train_x_aug_path = "./dau/fashion_harder/x_train_aug.npy"
    train_y_aug_path = "./dau/fashion_harder/y_train_aug.npy"
    train_x_ori_path = "./dau/fashion_harder/x_train_ori.npy"
    train_y_ori_path = "./dau/fashion_harder/y_train_ori.npy"
    test_x_aug_path = "./dau/fashion_harder/x_test_aug.npy"
    test_y_aug_path = "./dau/fashion_harder/y_test_aug.npy"
    test_x_ori_path = "./dau/fashion_harder/x_test_ori.npy"
    test_y_ori_path = "./dau/fashion_harder/y_test_ori.npy"

    train_x_aug_data = np.load(train_x_aug_path)
    train_y_aug_data = np.load(train_y_aug_path)
    train_x_ori_data = np.load(train_x_ori_path)
    train_y_ori_data = np.load(train_y_ori_path)
    test_x_aug_data = np.load(test_x_aug_path)
    test_y_aug_data = np.load(test_y_aug_path)
    test_x_ori_data = np.load(test_x_ori_path)
    test_y_ori_data = np.load(test_y_ori_path)

    # to select data set
    x_to_select_mnist, y_to_select_mnist = [], []
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    aug_or_ori = []  # aug: 1, ori: 0

    # 从 train_x_aug_data 和 train_x_ori_data 各选一半
    li = np.arange(len(train_x_ori_data))
    select_id = np.random.choice(a=li, size=10000, replace=False)
    print(len(select_id))

    for i in select_id:
        d1 = train_x_ori_data[i].reshape(1, 28, 28)
        d2 = train_x_aug_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(train_y_ori_data[i])
        aug_or_ori.append(0)
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(train_y_aug_data[i])
        aug_or_ori.append(1)
    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)
    aug_or_ori = np.array(aug_or_ori)
    state = np.random.get_state()
    np.random.shuffle(x_save_data)
    np.random.set_state(state)
    np.random.shuffle(y_save_data)
    np.random.set_state(state)
    np.random.shuffle(aug_or_ori)

    for k in range(len(test_x_ori_data)):
        dd1 = test_x_ori_data[k].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(test_y_ori_data[k])
        x_retrain_ori_test.append(dd1)
        y_retrain_ori_test.append(test_y_ori_data[k])

    for p in range(len(test_x_aug_data)):
        dd2 = test_x_aug_data[p].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(test_y_aug_data[p])
        x_retrain_aug_test.append(dd2)
        y_retrain_aug_test.append(test_y_aug_data[p])

    os.makedirs("./fashion_retrain", exist_ok=True)
    np.savez("./fashion_retrain/fashion_toselect", X=x_save_data, Y=y_save_data)
    np.save("./fashion_retrain/data_type", aug_or_ori)
    np.savez("./fashion_retrain/fashion_ori_test", X=x_retrain_ori_test, Y=y_retrain_ori_test)
    np.savez("./fashion_retrain/fashion_aug_test", X=x_retrain_aug_test, Y=y_retrain_aug_test)
    np.savez("./fashion_retrain/fashion_mix_test", X=x_retrain_mix_test, Y=y_retrain_mix_test)


def gen_mnist():
    train_x_aug_path = "./dau/mnist_harder/x_train_aug.npy"
    train_y_aug_path = "./dau/mnist_harder/y_train_aug.npy"
    train_x_ori_path = "./dau/mnist_harder/x_train_ori.npy"
    train_y_ori_path = "./dau/mnist_harder/y_train_ori.npy"
    test_x_aug_path = "./dau/mnist_harder/x_test_aug.npy"
    test_y_aug_path = "./dau/mnist_harder/y_test_aug.npy"
    test_x_ori_path = "./dau/mnist_harder/x_test_ori.npy"
    test_y_ori_path = "./dau/mnist_harder/y_test_ori.npy"

    train_x_aug_data = np.load(train_x_aug_path)
    train_y_aug_data = np.load(train_y_aug_path)
    train_x_ori_data = np.load(train_x_ori_path)
    train_y_ori_data = np.load(train_y_ori_path)
    test_x_aug_data = np.load(test_x_aug_path)
    test_y_aug_data = np.load(test_y_aug_path)
    test_x_ori_data = np.load(test_x_ori_path)
    test_y_ori_data = np.load(test_y_ori_path)

    # to select data set
    x_to_select_mnist, y_to_select_mnist = [], []
    x_retrain_ori_test, y_retrain_ori_test = [], []
    x_retrain_aug_test, y_retrain_aug_test = [], []
    x_retrain_mix_test, y_retrain_mix_test = [], []
    aug_or_ori = []  # aug: 1, ori: 0

    # 从 train_x_aug_data 和 train_x_ori_data 各选一半
    li = np.arange(len(train_x_ori_data))
    select_id = np.random.choice(a=li, size=10000, replace=False)
    print(len(select_id))

    for i in select_id:
        d1 = train_x_ori_data[i].reshape(1, 28, 28)
        d2 = train_x_aug_data[i].reshape(1, 28, 28)
        x_to_select_mnist.append(d1)
        y_to_select_mnist.append(train_y_ori_data[i])
        aug_or_ori.append(0)
        x_to_select_mnist.append(d2)
        y_to_select_mnist.append(train_y_aug_data[i])
        aug_or_ori.append(1)
    x_save_data = np.array(x_to_select_mnist)
    y_save_data = np.array(y_to_select_mnist)
    aug_or_ori = np.array(aug_or_ori)
    state = np.random.get_state()
    np.random.shuffle(x_save_data)
    np.random.set_state(state)
    np.random.shuffle(y_save_data)
    np.random.set_state(state)
    np.random.shuffle(aug_or_ori)

    for k in range(len(test_x_ori_data)):
        dd1 = test_x_ori_data[k].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd1)
        y_retrain_mix_test.append(test_y_ori_data[k])
        x_retrain_ori_test.append(dd1)
        y_retrain_ori_test.append(test_y_ori_data[k])

    for p in range(len(test_x_aug_data)):
        dd2 = test_x_aug_data[p].reshape(1, 28, 28)
        x_retrain_mix_test.append(dd2)
        y_retrain_mix_test.append(test_y_aug_data[p])
        x_retrain_aug_test.append(dd2)
        y_retrain_aug_test.append(test_y_aug_data[p])

    os.makedirs("./mnist_retrain", exist_ok=True)
    np.savez("./mnist_retrain/mnist_toselect", X=x_save_data, Y=y_save_data)
    np.save("./mnist_retrain/data_type", aug_or_ori)
    np.savez("./mnist_retrain/mnist_ori_test", X=x_retrain_ori_test, Y=y_retrain_ori_test)
    np.savez("./mnist_retrain/mnist_aug_test", X=x_retrain_aug_test, Y=y_retrain_aug_test)
    np.savez("./mnist_retrain/mnist_mix_test", X=x_retrain_mix_test, Y=y_retrain_mix_test)


if __name__ == '__main__':
    gen_fashion()
