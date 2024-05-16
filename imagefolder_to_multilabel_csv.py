import csv
import os
import pandas as pd
import numpy as np
import argparse

def flatten(l):
	return [item for sublist in l for item in sublist]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-root", type=str, default="")
	parser.add_argument("-csv_name", type=str, default="labels.csv")
	args = parser.parse_args()

	root = args.root
	csv_name = args.csv_name
	classes = os.listdir(root)

	imgs = flatten([[os.path.join(root, c, f) for f in os.listdir(
		os.path.join(root, c))] for c in classes])
	class_of_imgs = [p.split("/")[-2] for p in imgs]
	class_idxs = [classes.index(c) for c in class_of_imgs]

	eye = np.eye(len(classes))
	rows = np.array([eye[idx] for idx in class_idxs], dtype=int)

	df = pd.DataFrame(rows, columns=classes)
	df.insert(0, "filename", imgs)
	df.to_csv(csv_name, index=False)
