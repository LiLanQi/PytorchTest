import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt


register_matplotlib_converters()
sns.set_theme(style="whitegrid")

# rs = np.random.RandomState(365)
# values = rs.randn(365, 4).cumsum(axis=0)
# dates = pd.date_range("1 1 2016", periods=365, freq="D")
# data = pd.DataFrame(values, dates, columns=["ACE", "DMF", "NMF", "Water", "Meth"])
# data = data.rolling(7).mean()
data = pd.read_csv('D:/NMF_pred_label_data.csv')[0:1000]
# print(datas)
# plt.plot([0, 0], [-10000, -10000])
# sns.plot([3, 5], [1, 6],color="green")
# plt.plot([0, 5], [0, 5],color="red")
# plt.xlabel("Predicted Adsorption Energy")
# plt.ylabel("Actual Adsorption Energy")
# plt.title("Plot with 2 arbitrary Lines")
# plt.show()
sns.scatterplot(data=data, x="Predicted Adsorption Energy", y="Actual Adsorption Energy", size=0.01, x_bins=[-7000,-8000], y_bins=[-7000,-8000])
plt.plot([0, -10000], [0, -10000], linewidth=2)
plt.plot([500, -9500], [0, -10000], linewidth=2)
plt.plot([0, -10000], [500, -9500], linewidth=2)
plt.show()
