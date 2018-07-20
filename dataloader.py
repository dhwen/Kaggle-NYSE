import numpy as np
import csv


class DataLoader:

	def load(self, file_name, row_start, row_end):
		file_data = open(file_name,'r')

		data_reader = csv.reader(file_data)
		cur_row = 0;

		input = []
		labels = []

		stock_prev_val = 0;

		for row in data_reader:
			cur_row = cur_row + 1
			if (cur_row < row_start):
				continue
			elif (cur_row > row_end):
				break

			input.append(row[1:5])
			if(float(row[1:2][0]) > stock_prev_val):
				labels.append([0, 1])
			else:
				labels.append([1, 0])
			stock_prev_val = float(row[1:2][0])

		input.pop(len(input) - 1)
		labels.pop(0)

		return input, labels
