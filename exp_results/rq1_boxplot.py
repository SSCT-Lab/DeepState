import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def rq1_result_box(file_name):
    data = pd.read_csv(f'rq1/rq1_{file_name}.csv')
    labels = ['BTCov(CTM)', 'BSCov(CTM)', 'NC(CTM)', 'NC(CAM)', 'SC(CTM)', 'SC(CAM)',
                  'HSCov(CAM)', 'Random', 'DeepState']  # 图例
    ratio = [10, 20, 50]

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 4))
    plt.rcParams['pdf.use14corefonts'] = True

    for row, r in zip(axes, ratio):
        boxes = [data[f'Stellarbtcov{r}'] * 100, data[f'Stellarbscov{r}'] * 100, data[f'nc_ctm{r}'] * 100,
                 data[f'nc_cam{r}'] * 100, data[f'testRNNsc{r}'] * 100, data[f'testRNNsc_cam{r}'] * 100,
                 data[f'RNNTestcov{r}'] * 100, data[f'random{r}'] * 100, data[f'state_w{r}'] * 100]
        f = row.boxplot(boxes, labels=labels, vert=False, showmeans=False)
        row.grid(axis='y')
        # row.tick_params(axis='x', labelsize=18)
        row.set_yticklabels(labels, fontsize=20)
        row.tick_params(axis='x', labelsize=22)
        row.set_title(f"Selecting {r}% Tests", fontsize=23)

        for box in f['boxes']:
            box.set(linewidth=1.1)
        for whisker in f['whiskers']:
            whisker.set(linewidth=1.1)
        for cap in f['caps']:
            cap.set(linewidth=1.1)
        for median in f['medians']:
            median.set(linewidth=1.1)

    plt.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.05, hspace=0)  # 调整子图间距
    # plt.show()
    plt.savefig(f'./rq1-result-fig/rq1_{file_name}.pdf', dpi=200)


if __name__ == '__main__':
    file_names = ['mnist_lstm', 'mnist_blstm', 'fashion_lstm', 'fashion_gru', 'snips_blstm', 'snips_gru',
                  'agnews_lstm', 'agnews_blstm']
    for file_name in file_names:
        rq1_result_box(file_name)
