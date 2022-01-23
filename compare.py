import pandas as pd
from sklearn.metrics import accuracy_score


def compare(name):
    ann = pd.read_csv(f'./output/{name}.csv').Label.values
    ans = pd.read_csv('./output/sub.csv').Label.values
    print(accuracy_score(ans, ann))
