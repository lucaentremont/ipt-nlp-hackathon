import pandas as pd
import numpy as np
cleanedDataPath = "/Users/luca/Offline/data/clean.csv"
df = pd.read_csv(cleanedDataPath)
le = df.shape[0]
e = 1
for index, row in df.iterrows():
    print(index)
    mans = str(row['ManualGroups']).split('|')
    if len(mans) > 1:
        #print(mans)
        df.at[index, 'ManualGroups'] = mans[0]
        #print(row)
        for man in mans[1:]:
            print(man)
            #print(row2)
            df.loc[e+le] = row
            e+=1
            df.at[e+le-1,"ManualGroups"] = man
            print("REPLACING",e+le, "with",man )
df.to_csv("/Users/luca/Offline/data/expanded_manuals.csv", index=False)