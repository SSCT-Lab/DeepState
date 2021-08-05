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
    y = np.sort(x)[::-1]
    d = Counter(x)

    order = []
    for i in x:
        arg_li = np.where(y == i)[0]
        if len(arg_li) == 1:
            order.append(arg_li[0])
        elif len(arg_li) > 1:
            order.append(arg_li[d[i] - 1])
            d[i] = d[i] - 1
    return order


# The selection method of DeepState: change rate first, then compare the change trend.
def selection(change_rate_li, trend, n):
    d = Counter(change_rate_li)
    sorted_d = sorted(dict(d), reverse=True)  # The change rate is sorted from large to small, and count the numbers
    selected = np.zeros(len(change_rate_li))  # The selected mark is 1, and the eliminated mark is -1

    count = 0
    for value in sorted_d:
        num = dict(d)[value]  # The number of use cases corresponding to the current change rate
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
                    tmp_sim = calc_Jaccard_sim(tmp_trend1, tmp_trend2)  # The bigger the sim, the higher the similarity
                    # print("tmp_sim between case", place_li[j], "and", place_li[k], "is", tmp_sim)   #
                    if tmp_sim > 0.5:  # 0.2
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
    T_o = len(right)  # The size of the original sample
    T_s = collections_select[1]  # The size of the selected sample
    Tf_o = collections_right[0]  # The number of bug cases in the original sample
    Tf_s = 0  # The number of bug cases in the selected sample
    for right_value, select_value in zip(right, select):
        if right_value == 0 and select_value == 1:  # A bug case is detected and selected
            Tf_s += 1
    R = Tf_s / Tf_o if Tf_o != 0 else 0  # inclusiveness
    P = Tf_s / T_s if T_s != 0 else 0    # bug detection rate of the selected set
    O_P = Tf_o / T_o if T_o != 0 else 0  # bug detection rate of the original set

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


def cam_selection(x, length, select_num):
    selected = np.zeros(length)
    original_selected_num = len(x)
    if original_selected_num >= select_num:
        final_selected = x[:select_num]
    else:  # The use case selected by cov is smaller than the expected use case, then randomly add the remaining ones
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

