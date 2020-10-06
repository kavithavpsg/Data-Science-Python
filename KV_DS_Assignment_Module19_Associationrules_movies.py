# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 05:56:28 2020

@author: KavithaV
"""

# Implementing Apriori algorithm from mlxtend

# Conda install mlxtend

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

movies = pd.read_csv("D:/Data science/Module 19 - Association/Data Sets (1)/my_movies.csv")
cols = movies.columns[5:15]
movies.columns = movies.columns.str.replace(' ', '')
movies.columns
for c in cols:
    movies[c] = movies[c].apply(lambda x: np.nan if x == 0 else c)
movie_list =[] 
  
# Iterate over each row 
for index, rows in movies.iterrows(): 
    # Create list for the current row 
    
    my_list =[rows.V1,rows.V2,rows.V3,rows.V4,rows.V5,rows.SixthSense,rows.Gladiator, rows.LOTR1,rows.HarryPotter1,rows.Patriot, rows.LOTR2, rows.HarryPotter2, rows.LOTR,rows.Braveheart, rows.GreenMile] 
    #filter none values from the current row
    my_list = [i for i in my_list if i is not np.nan]
    movie_list.append(my_list) 
all_movie_list = [i for item in movie_list for i in item]

from collections import Counter
item_frequencies = Counter(all_movie_list)

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
movie_series  = pd.DataFrame(pd.Series(movie_list))

movie_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = movie_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

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

