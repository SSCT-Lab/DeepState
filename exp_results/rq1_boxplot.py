import pandas as pd
import matplotlib.pyplot as plt


def rq1_result_box():
    data = pd.read_csv("rq1/rq1_snips_blstm.csv")
    labels = ['BTCov(CTM)', 'BSCov(CTM)', 'NC(CTM)', 'NC(CAM)', 'SC(CTM)', 'SC(CAM)',
              'HSCov(CAM)', 'Random', 'DeepState']  # 图例

    ratio = [10, 20, 50]
    for r in ratio:
        plt.clf()
        plt.cla()
        plt.figure(figsize=(10, 4))  # 设置画布的尺寸
        plt.rcParams['pdf.use14corefonts'] = True

        boxes = [data[f'Stellarbtcov{r}'] * 100, data[f'Stellarbscov{r}'] * 100, data[f'nc_ctm{r}'] * 100,
                 data[f'nc_cam{r}'] * 100, data[f'testRNNsc{r}'] * 100, data[f'testRNNsc_cam{r}'] * 100,
                 data[f'RNNTestcov{r}'] * 100, data[f'random{r}'] * 100, data[f'state_w{r}'] * 100]

        plt.boxplot(boxes, labels=labels, vert=False, showmeans=False)
        plt.grid(axis='y')
        plt.ylabel("Selection Method", fontsize=21)
        plt.xlabel(f"Bug Detection Rate (%) with {r}% Tests Selected", fontsize=21)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # plt.show()
        plt.savefig(f"./rq1-result-fig/rq1_snips_blstm_{r}.pdf", dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    rq1_result_box()

