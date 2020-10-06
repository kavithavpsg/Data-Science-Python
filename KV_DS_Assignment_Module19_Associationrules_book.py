# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 06:18:28 2020

@author: KavithaV
"""

# Implementing Apriori algorithm from mlxtend

# Conda install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

book = pd.read_csv("D:/Data science/Module 19 - Association/Data Sets (1)/book.csv")
cols = book.columns
for c in book.columns:
    book[c] = book[c].apply(lambda x: None if x == 0 else c)
book_list =[] 
  
# Iterate over each row 
for index, rows in book.iterrows(): 
    # Create list for the current row 
    my_list =[rows.ChildBks, rows.YouthBks, rows.CookBks,rows.DoItYBks,rows.RefBks,rows.ArtBks,rows.GeogBks,rows.ItalCook,rows.ItalAtlas,rows.ItalArt,rows.Florence] 
    #filter none values from the current row
    my_list = [i for i in my_list if i is not None]
    book_list.append(my_list) 
all_book_list = [i for item in book_list for i in item]

from collections import Counter
item_frequencies = Counter(all_book_list)

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
book_series  = pd.DataFrame(pd.Series(book_list))

book_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = book_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

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

