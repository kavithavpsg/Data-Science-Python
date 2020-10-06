# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 06:41:09 2020

@author: KavithaV
"""

# Implementing Apriori algorithm from mlxtend

# Conda install mlxtend

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.corpus import stopwords
stop = stopwords.words('english')


retailtxns = []
colnames = ['Item1','Item2','Item3','Item4','Item5','Item6']
txns = pd.read_csv("D:/Data science/Module 19 - Association/Data Sets (1)/transactions_retail.csv",names=colnames,header=None)
txns = txns.dropna(axis = 0, how = 'any')
txns = txns.applymap(lambda s:s.lower() if type(s) == str else s)
###Preprocessing - to remove stop words from transactions
for i in stop :
    txns['Item1'] = txns['Item1'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
    txns['Item2'] = txns['Item2'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
    txns['Item3'] = txns['Item3'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
    txns['Item4'] = txns['Item4'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
    txns['Item5'] = txns['Item5'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
    txns['Item6'] = txns['Item6'].replace(to_replace=r'\b%s\b'%i, value=np.nan,regex=True)
txns = txns.dropna(axis = 0, how = 'any')
spec_chars_num = ["!",'"',"#","%","&","*","+",",","-",".","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–","0","1","2","3","4","5","6","7","8","9"]
txns.to_csv("D:/Data science/Module 19 - Association/Data Sets (1)/transactions_retail1.csv")    
with open("D:/Data science/Module 19 - Association/Data Sets (1)/transactions_retail1.csv") as f:
    retailtxns = f.read()

# splitting the data into separate transactions using separator as "\n"
retailtxns = retailtxns.split("\n")
retailtxns_list = []
for i in retailtxns:
    retailtxns_list.append(i.split(","))
    
all_retailtxns_list = [i for item in retailtxns_list for i in item]

len(retailtxns)

from collections import Counter
item_frequencies = Counter(retailtxns)

# After sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# Barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:10], x = list(range(0, 10)), color='rgbkymc'); plt.xticks(list(range(0,10),), items[0:10]); plt.xlabel("items"); plt.ylabel("Count")

# Creating Data Frame for the transactions data 

# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
retailtxns_series  = pd.DataFrame(pd.Series(retailtxns))

retailtxns_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retailtxns_series['transactions'].str.get_dummies()
frequent_itemsets = apriori(X, min_support=0.008, max_len=3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace=True)
plt.bar(x = list(range(1,11)), height = frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False, inplace=True)

help(rules.sort_values)

########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending=False).head(10)

