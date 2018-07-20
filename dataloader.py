import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import os
import shutil
import numpy as np
import csv


class DataLoader:

	def load(self, file_name):
		file_train = open(file_name,'r')

		train_reader = csv.reader(file_train)
		next(train_reader)

		train_input = []
		train_labels = []
		for row in train_reader:
			train_input.append(row[1:5])
			train_labels.append(row[1:2])

		train_input.pop(len(train_input) - 1)
		train_labels.pop(0)

		return train_input, train_labels
