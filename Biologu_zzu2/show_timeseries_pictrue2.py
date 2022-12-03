import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_theme(style="whitegrid")

# rs = np.random.RandomState(365)
# values = rs.randn(365, 4).cumsum(axis=0)
# dates = pd.date_range("1 1 2016", periods=365, freq="D")
# data = pd.DataFrame(values, dates, columns=["ACE", "DMF", "NMF", "Water", "Meth"])
# data = data.rolling(7).mean()
df = pd.read_csv('C:/Users/Administrator/Desktop/分子数据/ALL-LJ.csv')

print("data=", df)

sns.scatterplot(data=df)
plt.show()