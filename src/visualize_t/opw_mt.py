from algo import opw
import matplotlib.pyplot as plt
import seaborn as sns


def opw_viz(y1, y2, save_dir):
	dis, T = opw(y1.reshape(-1, 1), y2.reshape(-1, 1))

	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(T, ax=ax)
	plt.savefig(save_dir)
