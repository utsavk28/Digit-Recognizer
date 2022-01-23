import numpy as np
import matplotlib.pyplot as plt


def random_erasing(data, num=10):
    data1 = data.copy()
    data2 = data.copy()
    data3 = data.copy()
    for i in range(data.shape[0]):
        for _ in range(num):
            x, y = np.random.randint(0, 28), np.random.randint(0, 28)
            data1[i][x][y] = 0
            data2[i][x][y] = 1
            data3[i][x][y] = np.random.randint(1, 256) / 255.0
    data = np.concatenate((data1, data2))
    return data


def data_aug(data, y, num=10, seed=0):
    data1 = random_erasing(data, num)
    data1 = np.concatenate((data1, data))
    y = np.concatenate((y, y, y))
    return data1, y


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
