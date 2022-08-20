from algo import t_opw2
import matplotlib.pyplot as plt
import seaborn as sns


def t_opw2_viz(y1, y2, save_dir):
	dis, T = t_opw2(y1.reshape(-1, 1), y2.reshape(-1, 1))

	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(T, ax=ax)
	plt.savefig(save_dir)
