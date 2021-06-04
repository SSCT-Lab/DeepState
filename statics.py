from collections import Counter
import numpy as np
import pandas as pd
from deepstellar.coverage import Coverage


def mnist_input_preprocess(data):
    data = data.reshape(data.shape[0], 28, 28)
    data = data.astype('float32')
    data /= 255
    return data


# calculate change rate (without weights)
def cacl_change_rate(array):
    count = 0
    if len(array) <= 1:
        return 0
    for i in range(1, len(array)):
        if array[i] != array[i-1]:
            count = count + 1
    return count/(len(array)-1)


# calculate change rate (with weights)
def cacl_change_rate_with_weights(array):
    up = 0
    down = 0
    for i in range(1, len(array)):
        down = down + (i * i)
        if array[i] != array[i - 1]:
            up = up + (i * i)
    if down == 0:
        return 0
    else:
        return up/down


# get change set
def get_change_set(label_seq):
    change_set = set()
    for i in range(len(label_seq)-1):
        tmp1 = label_seq[i]
        tmp2 = label_seq[i+1]
        change_set.add(str(tmp1) + str(tmp2))
    return change_set


# calculate Jaccard similarity
def calc_Jaccard_sim(x, y):
    return len((x & y))/len((x | y)) if len((x | y)) != 0 else 0


def gini_sort_order(x):
    # x = np.array([0.14814815, 0.14814815, 0.2962963, 0.2962963, 0.8])
    y = np.sort(x)[::-1]
    # print("y:", y)
    d = Counter(x)

    order = []
    for i in x:
        arg_li = np.where(y == i)[0]
        # print(arg_li)
        if len(arg_li) == 1:
            order.append(arg_li[0])
        elif len(arg_li) > 1:
            order.append(arg_li[d[i] - 1])
            d[i] = d[i] - 1
    return order


# The selection method of DeepState: change rate first, then compare the change trend.
def selection(change_rate_li, trend, n):
    # x = np.array([0.14814815, 0.14814815, 0.2962963, 0.2962963, 0.8])
    # y = np.sort(x)[::-1]
    d = Counter(change_rate_li)
    sorted_d = sorted(dict(d), reverse=True)  # change rate从大到小排序，统计数量
    selected = np.zeros(len(change_rate_li))  # 选中的标记为1，剔除的标记为-1

    count = 0
    for value in sorted_d:
        # print("value", value)
        num = dict(d)[value]  # 当前change rate对应的用例数量
        if num == 1:
            place = np.where(change_rate_li == np.float64(value))[0][0]
            selected[place] = 1
            count += 1
            if count >= n:
                return selected

        elif num > 1:
            place_li = np.where(change_rate_li == np.float64(value))[0]
            for j in range(len(place_li)):
                if selected[place_li[j]] == -1 or selected[place_li[j]] == 1:
                    continue
                selected[place_li[j]] = 1
                count += 1
                if count >= n:
                    return selected

                tmp_trend1 = trend[place_li[j]]
                # print("selected case trend:", tmp_trend1)   #
                for k in range(j + 1, len(place_li)):
                    if selected[place_li[k]] == -1 or selected[place_li[k]] == 1:
                        continue
                    tmp_trend2 = trend[place_li[k]]
                    tmp_sim = calc_Jaccard_sim(tmp_trend1, tmp_trend2)  # sim越大，相似性越高
                    # print("tmp_sim between case", place_li[j], "and", place_li[k], "is", tmp_sim)   #
                    if tmp_sim > 0.5:  # 0.5
                        selected[place_li[k]] = -1
                    # else:
                    #     selected[place_li[k]] = 1
                    #     count += 1
                    #     if count >= n:
                    #         return selected

    if count < n:
        print("selection not enough. It will full fill the other cases.")
        for p in range(len(selected)):
            if selected[p] == -1:
                selected[p] = 1
                count += 1
                if count == n:
                    return selected


def ran_selection(length, select_num):
    x = np.zeros(length-select_num)
    y = np.ones(select_num)
    z = np.concatenate((x, y))
    np.random.shuffle(z)
    return z


# selection evaluation
def selection_evaluate(right, select):
    collections_right = Counter(right)
    collections_select = Counter(select)
    T_o = len(right)  # 原始样本的大小
    T_s = collections_select[1]  # 选择样本的大小
    Tf_o = collections_right[0]  # 原始样本里的bug用例个数
    Tf_s = 0  # 选择集里的bug用例个数
    for right_value, select_value in zip(right, select):
        if right_value == 0 and select_value == 1:  # 检测出bug且被选中
            Tf_s += 1
    R = Tf_s / Tf_o if Tf_o != 0 else 0  # 查全率
    P = Tf_s / T_s if T_s != 0 else 0    # 准确率，选择集的bug检测率
    O_P = Tf_o / T_o if T_o != 0 else 0  # 原始bug检测率

    theo_R = T_s / Tf_o if T_s < Tf_o else 1
    theo_P = Tf_o / T_s if T_s > Tf_o else 1

    return R, P, O_P, theo_R, theo_P


# check the predict result the right or wrong
def check_predict_result(predict, label, right):
    if predict == label:
        # print("predict right:", 1)
        right.append(1)
    else:
        # print("predict right:", 0)
        right.append(0)


def cam_sort_order(x, length):
    y = np.array(range(length))
    z = np.setdiff1d(y, x)
    return np.append(x, z)


def cam_selection(x, length, select_num):
    selected = np.zeros(length)
    original_selected_num = len(x)
    if original_selected_num >= select_num:
        final_selected = x[:select_num]
    else:  # cov选出来的用例小于期望选出的用例，随机补上剩下的
        tmp = np.setdiff1d(np.arange(length), x)
        np.random.shuffle(tmp)
        final_selected = np.append(x, tmp[:(select_num-original_selected_num)])
    # print(final_selected)
    for i in final_selected:
        selected[i] = 1
    return selected


def ctm_selection(cov, length, selected_num):
    selected = np.zeros(length)
    arg_sorted_cov = cov.argsort()[::-1]
    for i in arg_sorted_cov[:selected_num]:
        selected[i] = 1
    return selected


def nc_cam_selection(nc_cam, length, select_num):
    final_selected = np.zeros(length)
    selected_id = []

    count = 0
    for i in range(len(nc_cam)):
        if nc_cam[i] == 1:
            selected_id.append(i)
            count += 1
            if count >= select_num:
                break
    if count < select_num:
        tmp = np.setdiff1d(np.arange(length), selected_id)
        np.random.shuffle(tmp)
        selected_id = selected_id + list(tmp[:(select_num - count)])

    for i in selected_id:
        final_selected[i] = 1

    return final_selected


def gini_selection(gini, length, selected_num):
    selected = np.zeros(length)
    arg_sorted_gini = gini.argsort()[::-1]
    for i in arg_sorted_gini[:selected_num]:
        selected[i] = 1
    return selected


def ran_order(length):
    x = np.arange(length)
    x = np.asarray(x)
    np.random.shuffle(x)
    return x


def get_stellar_cov(classifier, model, x, dtmc_wrapper_f):
    BSCov, BTCov = 0, 0
    stats = classifier.get_state_profile(np.array([x]), model)
    coverage_handlers = []

    for criteria, k_step in [("state", 0), ("transition", 0)]:  # , ("k-step", 3), ("k-step", 6)
        cov = Coverage(dtmc_wrapper_f, criteria, k_step)
        coverage_handlers.append(cov)

    for i, coverage_handler in enumerate(coverage_handlers):
        cov = coverage_handler.get_coverage_criteria(stats)
        total = coverage_handler.get_total()
        if i == 0:
            BSCov = len(cov) / total  # Basic State Coverage(BSCov)
        if i == 1:
            BTCov = len(cov) / total  # Basic Transition Coverage(BTCov)
    return BSCov, BTCov


def get_testrnn_sc(plus_sum, minus_sum):
    count = 0
    act_time = []
    for i in range(1, len(plus_sum)):
        delta = abs(plus_sum[i]-plus_sum[i-1]) + abs(minus_sum[i]-minus_sum[i-1])
        if delta >= 0.6:
            count += 1
            act_time.append(i)
    sc = count / len(plus_sum) if count != 0 else 0
    return sc, set(act_time)


def get_nc_activate(lstm_out):
    activated = np.argwhere(lstm_out[0] > 0).tolist()
    activated_li = []
    for a in activated:
        a = tuple(a)
        activated_li.append(a)
    act = set(activated_li)
    return act


def deepstate_selection_for_sort(change_rate_li, trend, n):
    sort_li = []
    d = Counter(change_rate_li)
    sorted_d = sorted(dict(d), reverse=True)  # change rate从大到小排序，统计数量
    selected = np.zeros(len(change_rate_li))  # 选中的标记为1，剔除的标记为-1

    count = 0
    for value in sorted_d:
        # print("value", value)
        num = dict(d)[value]  # 当前change rate对应的用例数量
        if num == 1:
            place = np.where(change_rate_li == np.float64(value))[0][0]
            selected[place] = 1
            count += 1
            sort_li.append(place)
            if count >= n:
                return sort_li

        elif num > 1:
            place_li = np.where(change_rate_li == np.float64(value))[0]
            for j in range(len(place_li)):
                if selected[place_li[j]] == -1 or selected[place_li[j]] == 1:
                    continue
                selected[place_li[j]] = 1
                count += 1
                sort_li.append(place_li[j])
                if count >= n:
                    return sort_li

                tmp_trend1 = trend[place_li[j]]
                # print("selected case trend:", tmp_trend1)   #
                for k in range(j + 1, len(place_li)):
                    if selected[place_li[k]] == -1 or selected[place_li[k]] == 1:
                        continue
                    tmp_trend2 = trend[place_li[k]]
                    tmp_sim = calc_Jaccard_sim(tmp_trend1, tmp_trend2)  # sim越大，相似性越高
                    # print("tmp_sim between case", place_li[j], "and", place_li[k], "is", tmp_sim)   #
                    if tmp_sim > 0.5:  # 0.5
                        selected[place_li[k]] = -1

    if count < n:
        # print("selection not enough. It will full fill the other cases.")
        for p in range(len(selected)):
            if selected[p] == -1:
                selected[p] = 1
                count += 1
                sort_li.append(p)
                if count == n:
                    return sort_li


def deepstate_sort(length, sort_li):
    order_li = np.zeros(length)
    for idx, num in enumerate(sort_li):
        order_li[num] = idx
    return order_li


def ctm_sort_order(cov, length, selected_num):
    sort_li = []
    arg_sorted_cov = cov.argsort()[::-1]
    for i in arg_sorted_cov[:selected_num]:
        sort_li.append(i)
    order_li = np.zeros(length)
    for idx, num in enumerate(sort_li):
        order_li[num] = idx
    return order_li


def apfd1(right, sort):
    length = np.sum(sort != 0)
    if length != len(sort):
        sort[sort == 0] = np.random.permutation(len(sort) - length) + length + 1
    sum_all = np.sum(sort.values[[right.values != 1]])
    n = len(sort)
    m = pd.value_counts(right)[0]
    return 1 - float(sum_all)/(n*m)+1./(2*n)


# 归一化了的APFD
def apfd(right, sort):
    length = np.sum(sort != 0)
    #     print(length)
    #     print(len(sort))
    if length != len(sort):
        np.random.seed(42)
        sort[sort == 0] = np.random.permutation(len(sort) - length) + length + 1  # 随机选择,所以cam的结果未必一致
    sum_all = np.sum(sort.values[(right.values != 1)])
    #  print((right.values != 1))
    n = len(sort)
    m = pd.value_counts(right)[0]
    #  print(len(sort.values[(right.values != 1)]), pd.value_counts(right)[0])
    # 归一化
    sum_min = sum(list(range(1, len(sort.values[(right.values != 1)]) + 1)))
    sum_max = sum(list(range(len(sort), len(sort) - len(sort.values[(right.values != 1)]), -1)))

    #  print(sum_min)
    res = 1 - float(sum_all) / (n * m) + 1. / (2 * n)
    res_max = 1 - float(sum_min) / (n * m) + 1. / (2 * n)
    res_min = 1 - float(sum_max) / (n * m) + 1. / (2 * n)
    #  print(res_max)
    #  return res / res_max
    return (res - res_min) / (res_max - res_min)


def get_apfd(result_path):
    data_dict = pd.read_csv(result_path, index_col=0)
    print('==============================')
    print('APFD of State:{}'.format(apfd(data_dict.right, data_dict.state)))
    print('APFD of W_State:{}'.format(apfd(data_dict.right, data_dict.w_state)))
    print('APFD of State_dir:{}'.format(apfd(data_dict.right, data_dict.state_dir)))
    print('APFD of W_State_dir:{}'.format(apfd(data_dict.right, data_dict.w_state_dir)))
    print('APFD of Gini:{}'.format(apfd(data_dict.right, data_dict.gini)))
    print('APFD of hs_cov:{}'.format(apfd(data_dict.right, data_dict.hs_cov)))
    print('APFD of random:{}'.format(apfd(data_dict.right, data_dict.random)))
    print('APFD of bscov:{}'.format(apfd(data_dict.right, data_dict.bscov)))
    print('APFD of btcov:{}'.format(apfd(data_dict.right, data_dict.btcov)))
    print('==============================')


if __name__ == '__main__':
    data_dict = {}
    data_dict["seed"] = pd.read_csv("order_result0514.csv", index_col=0)
    for key in data_dict.keys():
        print(key)
        print('APFD of State:{}'.format(apfd(data_dict[key].right, data_dict[key].state)))
        print('APFD of W_State:{}'.format(apfd(data_dict[key].right, data_dict[key].w_state)))
        # print('APFD of Gini:{}'.format(apfd(data_dict[key].right, data_dict[key].gini)))
        print('APFD of hs_cov:{}'.format(apfd(data_dict[key].right, data_dict[key].hs_cov)))
        print('APFD of random:{}'.format(apfd(data_dict[key].right, data_dict[key].random)))
        print('==============================')
