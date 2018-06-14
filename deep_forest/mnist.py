import argparse
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
	
	# config
    args = parse_args()
    if args.model == 'ca':
        config = load_json('./mnist-ca.json')
    elif args.model == 'gc':
        config = load_json('./mnist-gc.json')
    else:
        config = load_json('./mnist-gc.json')

    gc = GCForest(config)
	
    # gc.set_keep_model_in_mem(False)
    gc.set_keep_model_in_mem(True)


    # data
    data_num_train = 60000  # The number of figures
    data_num_test = 10000 # test num
    fig_w = 45  # width of each figure

    X_train = np.fromfile("./data/mnist_train/mnist_train_data", dtype=np.uint8)
    y_train = np.fromfile("./data/mnist_train/mnist_train_label", dtype=np.uint8)
    X_test= np.fromfile("./data/mnist_test/mnist_test_data", dtype=np.uint8)
    y_test= np.fromfile("./data/mnist_test/mnist_test_label", dtype=np.uint8)
	
    # reshape the matrix
    X_train = X_train.reshape(data_num_train,1, fig_w, fig_w)
    X_test= X_test.reshape(data_num_test, 1,fig_w, fig_w)


    # train
    X_train_enc, X_test_enc = gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test)

	# test
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))


