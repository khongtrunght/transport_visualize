import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import altair as alt
import os

os.chdir('../')


# generate normal distribution
def create_syn_data():
	var_1 = 5
	var_2 = 10
	norm1 = norm(loc=31, scale=var_1)
	norm2 = norm(loc=51, scale=var_2)

	norm1.pdf(31) * var_1 * (np.sqrt(2 * np.pi))

	height = 1

	x = np.arange(0, 100, 1)
	scale1 = height * var_1 * (np.sqrt(2 * np.pi))
	scale2 = height * var_2 * (np.sqrt(2 * np.pi))
	y1 = 0.1 + scale1 * norm1.pdf(x)
	y2 = 0.1 + scale2 * norm2.pdf(x)

	data = pd.DataFrame({"x": x, "y1": y1, "y2": y2})
	return data


if __name__ == '__main__':
	data = create_syn_data()

	chart_1 = alt.Chart(data).mark_line().encode(
		x=alt.X("x", title="Time Point"),
		y=alt.Y("y1", scale=alt.Scale(domain=(0, 1.2)), axis=alt.Axis(tickCount=5, title="Value")),
		color=alt.value("blue"),
		strokeWidth=alt.value(1)
	)

	chart_2 = alt.Chart(data).mark_line().encode(
		x="x",
		y=alt.Y("y2", scale=alt.Scale(domain=(0, 1.2))),
		color=alt.value("orange"),
		strokeWidth=alt.value(1)
	)

	# disable grid , lower tick
	chart = (chart_1 + chart_2).properties(width=500, height=400).configure_axis(grid=False)
	chart.save("imgs/chart.html")

	# if folder data not exist, create it
	if not os.path.exists("data"):
		os.makedirs("data")

	data.to_csv("data/data.csv", index=False)
# print(data.head())
