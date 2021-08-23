import pandas as pd
import numpy as np


if __name__ == '__main__':
    data = pd.read_csv("rq1/rq1_agnews_blstm.csv")
    r = 10

    boxes = [data[f'state_w{r}'], data[f'random{r}'], data[f'RNNTestcov{r}'],
             data[f'testRNNsc_cam{r}'], data[f'testRNNsc{r}'], data[f'nc_cam{r}'],
             data[f'nc_ctm{r}'], data[f'Stellarbscov{r}'], data[f'Stellarbtcov{r}']]

    for i in range(9):
        print(np.around(np.mean(boxes[i]), 2))
