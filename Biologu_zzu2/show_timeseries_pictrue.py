# Import Data
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('C:/Users/Administrator/Desktop/分子数据/ALL-LJ.csv')
print(df.head)
# Define the upper limit, lower limit, interval of Y axis and colors
y_LL = int(df.iloc[:, 1:].min().min()*1.1)
y_UL = int(df.iloc[:, 1:].max().max()*1.1)
print("y_UL=", y_UL)
print("y_LL=", y_LL)
# print(df.iloc[:, 1:].max().max())
y_interval = 2000
mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink']

# Draw Plot and Annotate
fig, ax = plt.subplots(1,1,figsize=(16, 9), dpi= 80)

columns = df.columns[0:]
columns = columns[1:]
print(columns)
for i, column in enumerate(columns):
    print("i=", i, "column=", column)
    plt.plot(df["indexx"].values, df[column].values, lw=1.5, color=mycolors[i])
    if (i == 1):
        plt.text(df.shape[0] + 1000, df[column].values[-1] - 500, column, fontsize=14, color=mycolors[i])  # 标签添加的位置
    else:
        plt.text(df.shape[0]+1000, df[column].values[-1]+100, column, fontsize=14, color=mycolors[i]) #标签添加的位置

# Draw Tick lines
for y in range(y_LL, y_UL, y_interval):
    plt.hlines(y, xmin=0, xmax=100000, colors='black', alpha=0.3, linestyles="--", lw=0.5)

# Decorations
plt.tick_params(axis="both", which="both", bottom=False, top=False,
                labelbottom=True, left=False, right=False, labelleft=True)
x_index = [0, 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
# Lighten borders
plt.gca().spines["top"].set_alpha(.3)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.3)
plt.gca().spines["left"].set_alpha(.3)
print("df.shape[0]=", df.shape[0])
for i in range(0, df.shape[0], 10000):
    print("i=", i)
plt.title('Changes in the energy of the system at different times', fontsize=20)
plt.yticks(range(y_LL, y_UL, y_interval), [str(y) for y in range(y_LL, y_UL, y_interval)], fontsize=12)
plt.xticks(range(0, df.shape[0], 10000), [str(x) for x in range(0, df.shape[0], 10000)],  fontsize=12)
plt.ylim(y_LL, y_UL)
plt.xlim(-2, 120000)
plt.show()