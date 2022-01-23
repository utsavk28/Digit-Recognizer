import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from compare import compare
from utils import plot_history
from models import build_ann_model

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')

X = train.drop(columns=['label'], axis=1)
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000, random_state=42, stratify=y)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
ans = pd.read_csv('./input/sample_submission.csv')
fans = None
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(fold)
    train_x, train_y = X_train.loc[train_idx], y_train.loc[train_idx]
    val_x, val_y = X_train.loc[val_idx], y_train.loc[val_idx]
    print(train_x.shape, train_y.shape)

    model = build_ann_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.compile(optimizer=Adam(learning_rate=5e-5)
                  , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1000, batch_size=128, verbose=0,
                        callbacks=[callback])
    print(callback.stopped_epoch)
    model.evaluate(train_x, train_y)
    model.evaluate(X_test, y_test)
    pred_val = np.argmax(model.predict(val_x), axis=-1)
    pred_test = np.argmax(model.predict(X_test), axis=-1)
    test_score = accuracy_score(y_test.values, pred_test, )
    models.append((fold, model, accuracy_score(train_y.values, np.argmax(model.predict(train_x), axis=-1)),
                   accuracy_score(val_y.values, pred_val), test_score))

    pred_test = np.argmax(model.predict(test), axis=-1)
    if fans is None:
        ans['Label'] = pred_test
        fans = test_score
    else:
        if test_score > fans:
            ans['Label'] = pred_test
            fans = test_score
    plot_history(history)

print(models)
ans.to_csv('./output/sub_ann.csv', index=False)
compare('sub_ann')
