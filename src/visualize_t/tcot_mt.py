from algo import tcot
import matplotlib.pyplot as plt
import seaborn as sns


def tcot_viz(y1, y2, save_dir):
	dis, T = tcot(y1.reshape(-1, 1), y2.reshape(-1, 1))

	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(T, ax=ax)
	plt.xlabel("Y2")
	plt.ylabel("Y1")
	plt.savefig(save_dir)