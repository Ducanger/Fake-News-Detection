import pandas as pd
import re

df = pd.read_csv(r'D:\Dat\Fake News Detection\Data\pre-train-data.csv') 

count1 = 0
for i in range (0, len(df)):
    if df.label[i] == 1: count1 = count1 + 1

data = {'post_message': ['Test'],
	    'label': [0],}
df_export = pd.DataFrame(data)

count0 = 0

for i in range (0, len(df)):
    if df.label[i] == 0: 
        count0 = count0 + 1
        if count0 <= 1000: 
            new_row = {'post_message': df.post_message[i], 'label': df.label[i]}
            df_export = df_export.append(new_row, ignore_index=True)
    else:
        new_row = {'post_message': df.post_message[i], 'label': df.label[i]}
        df_export = df_export.append(new_row, ignore_index=True)
	
df_export = df_export.drop(df_export.index[0])
print(len(df_export))

# export file
df_export.to_csv(r"D:\Dat\Fake News Detection\Data\dataBERT.csv")
