import numpy as np
import pandas as pd


def viz_image(arr):
    arr = np.array(arr)
    arr = arr.reshape(28, 28)
    img = ""
    for i in range(28):
        for j in range(28):
            arr[i][j] = 1 if arr[i][j] > 0 else 0
            img += str(arr[i][j])
        img += '\n'

    return img


train = pd.read_csv('input/train.csv')
print(train.head())
for i in range(10):
    print(train.iloc[i, 0])
    print(viz_image(train.iloc[i, 1:].values.tolist()))
