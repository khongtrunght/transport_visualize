from create_data import create_syn_data
import os
from visualize_t import *



data = create_syn_data()
if not os.path.exists("data"):
	os.makedirs("data")

data.to_csv("data/data.csv", index=False)

y1 = data["y1"].values
y2 = data["y2"].values




img_dir = "imgs"

dtw_viz(y1, y2, os.path.join(img_dir,"dtw.jpg"))
soft_dtw_viz(y1, y2, os.path.join(img_dir, "soft_dtw.jpg"))
opw_viz(y1, y2, os.path.join(img_dir,"opw.jpg"))
t_opw1_viz(y1, y2, os.path.join(img_dir,"t_opw1.jpg"))
t_opw2_viz(y1, y2, os.path.join(img_dir,"t_opw2.jpg"))
tcot_viz(y1, y2, os.path.join(img_dir,"tcot.jpg"))