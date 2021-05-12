import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import numpy as np


def prepare_dataset_for_dense(train_data_path, classes_num):
	df_r = pd.read_csv(train_data_path)
	inputs_num = len(list(df_r.iloc[0])) - classes_num
	df = df_r.sample(frac=1)
	y_df = df.iloc[:, inputs_num:]
	x_df = df.iloc[:, :inputs_num]

	y_number = []
	for i in range(len(y_df)):
		y_array = list(y_df.iloc[i])
		y_number.append(np.argmax(y_array))

	y = utils.to_categorical(y_number, classes_num)

	return x_df, y, inputs_num


