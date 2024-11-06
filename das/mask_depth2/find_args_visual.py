# encoding=UTF8
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from scipy import stats

all_num = []
all_distance = []

with open(r'output_detailed.csv', 'r') as f:
    reader = csv.reader(f)
    print(type(reader))
    for row in reader:
        for ele in list(filter(None, row)):
            if ele is None or ele == "":
                continue
            res = list(filter(None, ele[1:-1].split(' ')))
            # print(len(res))
            print(res)
            if(len(res)> 1):
                all_num.append(float(res[0]))
                all_distance.append(float(res[1]))


#
# plt.scatter(all_num,
#             all_distance,
#             c='red',
#             label='function')
all_num = np.array(all_num)
all_distance = np.array(all_distance)

model1 = np. poly1d (np. polyfit (all_num, all_distance, 1))
model2 = np. poly1d (np. polyfit (all_num, all_distance, 2))
model3 = np. poly1d (np. polyfit (all_num , all_distance, 3))
model4 = np. poly1d (np. polyfit (all_num , all_distance, 4))
model5 = np. poly1d (np. polyfit (all_num, all_distance, 5))

polyline = np. linspace (int(min(all_distance)), int(max(all_distance)), 50 )
print(polyline)

plt. plot (polyline, model1(polyline), color='green')
plt. plot (polyline, model2(polyline), color='red')
plt. plot (polyline, model3(polyline), color='purple')
plt. plot (polyline, model4(polyline), color='blue')
plt. plot (polyline, model5(polyline), color='orange')

# print(len(all_num))
# print(len(all_distance))
# plt.plot(all_num,all_distance)
#
# # plt.legend()  # 显示图例
#
# plt.show(block=True)  # 显示所绘图形

# all_num = [4., 2., 2.]
# all_distance = [42.48598918, 81.04700026, 54.76872127]

# plt.figure()
plt.scatter(all_num, all_distance)
# plt.text(cmax * 0.1, cmax * 0.9, 'R$^2$=%s' % (np.around(r_value ** 2, 2)), fontsize=20)
# plt.text(cmax * 0.1, cmax * 0.8, 'y=%s*x+%s' % (np.around(slope, 2), np.around(intercept, 2)), fontsize=15)


# x_major_locator=MultipleLocator(50)
# y_major_locator = MultipleLocator(100)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

# plt.xlim(0,600)
# plt.ylim(0,2000)

plt.show()