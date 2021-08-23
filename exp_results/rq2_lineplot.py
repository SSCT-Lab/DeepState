import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def line_chart(data_path):
    data = pd.read_csv(f"rq2/rq2_{data_path}.csv")

    datas = [data['random']*100, data['state']*100, data['RNNTestcov']*100, data['Stellarbscov']*100,
             data['Stellarbtcov']*100, data['testRNNsc']*100, data['testRNNsc_cam']*100, data['nc_ctm']*100,
             data['nc_cam']*100]
    labels = ['Random', 'DeepState', 'RNNTest-HSCov(CAM)', 'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)',
              'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)']  # labels
    x = np.arange(1, 41)

    plt.figure(figsize=(10, 7))
    plt.rcParams['pdf.use14corefonts'] = True

    plt.plot(x, datas[0][1:41], label=labels[0], color='black', linewidth=4)
    plt.plot(x, datas[1][1:41], label=labels[1], color='red', linewidth=4)
    for i in range(2, 9):
        plt.plot(x, datas[i][1:41], label=labels[i])

    plt.legend(fontsize=19)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.xlabel('Selection Rate', fontsize=24)
    plt.ylabel('Inclusiveness', fontsize=24)

    # plt.show()
    plt.savefig(f"./rq2-result-fig/rq2_{data_path}.pdf", dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    file_names = ['mnist_lstm', 'mnist_blstm', 'fashion_lstm', 'fashion_gru', 'snips_blstm', 'snips_gru',
                  'agnews_lstm', 'agnews_blstm']
    for file_name in file_names:
        line_chart(file_name)


