from dtw import *
import seaborn as sns
from scipy.sparse import coo_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os


def dtw_viz(y1, y2, save_dir):
	alignment = dtw(y1, y2, keep_internals=True)
	coor = np.vstack((alignment.index1, alignment.index2))
	sparse_matrix = coo_matrix((np.ones_like(alignment.index1), coor)).toarray()

	fig, ax = plt.subplots(figsize=(8, 6))

	ax.invert_yaxis()
	sns.heatmap(sparse_matrix, ax=ax)

	plt.savefig(save_dir)
