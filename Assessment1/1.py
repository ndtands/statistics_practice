cocacola_normal = [371,370,370,373,374,372,375,371]
cocacola_diet   = [353,355,352,354,355,356,355,357]

import statistics as lib
from numpy import quantile
import pandas as pd
print("Coca-cola thường")
print("  >> Giá trị trung bình: ",lib.median(cocacola_normal))
print("  >> Giá trị trung vị: ",lib.mean(cocacola_normal))
print("  >> Giá trị độ lệch chuẩn: ",lib.stdev(cocacola_normal))
print("  >> Giá trị phương sai: ",lib.variance(cocacola_normal))
df_cocacola_normal = pd.DataFrame(cocacola_normal, columns=['val'])
quantile_cocacola_normal=df_cocacola_normal.quantile([0.0,.25,.5,.75,1.0])['val']
print("  >> Q1 , Q2 , Q3: ",quantile_cocacola_normal[0.25],",",quantile_cocacola_normal[0.5],",",quantile_cocacola_normal[0.75])
print("  >> Biên độ: ",quantile_cocacola_normal[1.0]-quantile_cocacola_normal[0.0])
IQR=quantile_cocacola_normal[0.75]-quantile_cocacola_normal[0.25]
print("  >> IQR: ",IQR)
min_out=quantile_cocacola_normal[0.25]-1.5*IQR
max_out=quantile_cocacola_normal[0.75]+1.5*IQR
print("  >> Khoảng giới hạn outline: ","[",quantile_cocacola_normal[0.25]-1.5*IQR,':',quantile_cocacola_normal[0.75]+1.5*IQR,"]")
outline=[]
for i in cocacola_normal:
    if min_out > i or i >max_out:
        outline.append(i)
print("  >> Các outline: ",None if len(outline)==0 else outline)

print("Coca-cola ăn kiên")
print("  >> Giá trị trung bình: ",lib.median(cocacola_diet))
print("  >> Giá trị trung vị: ",lib.mean(cocacola_diet))
print("  >> Giá trị độ lệch chuẩn: ",lib.stdev(cocacola_diet))
print("  >> Giá trị phương sai: ",lib.variance(cocacola_diet))
df_cocacola_diet = pd.DataFrame(cocacola_diet, columns=['val'])
quantile_cocacola_diet=df_cocacola_diet.quantile([0.0,.25,.5,.75,1.0])['val']
print("  >> Q1 , Q2 , Q3: ",quantile_cocacola_diet[0.25],",",quantile_cocacola_diet[0.5],",",quantile_cocacola_diet[0.75])
print("  >> Biên độ: ",quantile_cocacola_diet[1.0]-quantile_cocacola_diet[0.0])
IQR=quantile_cocacola_diet[0.75]-quantile_cocacola_diet[0.25]
print("  >> IQR: ",IQR)
min_out=quantile_cocacola_diet[0.25]-1.5*IQR
max_out=quantile_cocacola_diet[0.75]+1.5*IQR
print("  >> Khoảng giới hạn outline: ","[",quantile_cocacola_diet[0.25]-1.5*IQR,':',quantile_cocacola_diet[0.75]+1.5*IQR,"]")
outline=[]
for i in cocacola_diet:
    if min_out > i or i >max_out:
        outline.append(i)
print("  >> Các outline: ",None if len(outline)==0 else outline)

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
data_1 = np.array(cocacola_normal) 
data_2 =np.array(cocacola_diet)
data =[data_1,data_2]

fig = plt.figure(figsize =(10, 7))
# Creating plot
plt.boxplot(data)
 
# show plot
plt.show()