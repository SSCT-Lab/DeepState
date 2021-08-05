import pandas as pd
import re
from nltk.tokenize import word_tokenize


# 原始数据集是intent_data.csv, 剔除单词个数小于6和大于16的之后, 生成new_intent.csv 用于模型训练和预测
if __name__ == '__main__':
    data_path = "intent_data.csv"
    data = pd.read_csv(data_path)
    drop_li = []
    for i in range(len(data)):
        text = data.text[i]
        print(text)

        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        w = word_tokenize(clean)
        words = [i.lower() for i in w]

        print(words)
        print(len(words))

        if len(words) < 6 or len(words) > 16:
            drop_li.append(i)
    print(drop_li)
    print(len(drop_li))

    new_data = data.drop(drop_li)
    new_data.to_csv("new_intent.csv", index=0)
