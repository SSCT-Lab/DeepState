import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def line_chart():
    data = pd.read_csv("rq3/line_rq3_mnist_lstm.csv")
    mix = [data['random_selected_mix'], data['state_w_selected_mix'], data['cov_selected_mix'],
                 data['bscov_selected_mix'], data['btcov_selected_mix'],
                 data['sc_ctm_selected_mix'], data['sc_cam_selected_mix'],
                 data['nc_ctm_selected_mix'], data['nc_cam_selected_mix']]

    aug = [data['random_selected_aug'], data['state_w_selected_aug'],
             data['cov_selected_aug'],
             data['bscov_selected_aug'], data['btcov_selected_aug'],
             data['sc_ctm_selected_aug'], data['sc_cam_selected_aug'],
             data['nc_ctm_selected_aug'], data['nc_cam_selected_aug']]

    labels = ['Random', 'DeepState', 'RNNTest-HSCov(CAM)', 'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)',
              'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)', ]  # 图例

    x = data['select rate']

    data_types = ['mix', 'aug']

    for data_type in data_types:
        plt.figure(figsize=(9, 6))
        plt.rcParams['pdf.use14corefonts'] = True

        plt.plot(x, eval(data_type)[0] * 0.01, label=labels[0], color='black', linewidth=3)
        plt.plot(x, eval(data_type)[1] * 0.01, label=labels[1], color='red', linewidth=3)

        for i in range(2, 9):
            plt.plot(x, eval(data_type)[i] * 0.01, label=labels[i])

        plt.legend(fontsize=12)
        plt.xticks(x, [i for i in x], fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Selection rate', fontsize=19)
        plt.ylabel('Acc Imp (aug)', fontsize=19)

        # plt.show()
        plt.savefig(f"./rq3-result-fig/rq3_mnist_lstm_{data_type}.pdf", dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    line_chart()
