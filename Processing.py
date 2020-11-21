import pandas as pd
import numpy as np
from copy import deepcopy

rating_path = 'C:\Users\zhuof\Desktop\CourseProjects\RecommendationSystem'
title_path = 'C:\Users\zhuof\Desktop\CourseProjects\RecommendationSystem'
ratings = pd.read_csv(rating_path, sep  = "::", names = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
titles = pd.read_csv(title_path, sep = "::", names = ['MovieID', 'Title', 'Genres'], encoding ='latin-1')
data = pd.merge(ratings, titles, on='MovieID')
num_data = data.shape[0]
num_total = data.shape[0]
num_train = int(np.floor(num_total*0.8))
num_test = num_total - num_train

# split train data, valid data, test data.
data = data.sample(frac = 1)
train_data = data.head(0.8 * num_total)
val_data = train_data.tail(0.25 * num_train)
test_data = data.tail(0.2 * num_total)
train_mtx = train_data.pivot_table(index='UserID', columns='Title', values='Rating').values
val_mtx = val_data.pivot_table(index='UserID', columns='Title', values='Rating').values
test_mtx = test_data.pivot_table(index='UserID', columns='Title', values='Rating').values

def select(input_Matrix):
    train_select = []
    for i, row in enumerate(input_Matrix):
        for j, entry in enumerate(row):
            if input_Matrix[i][j] != np.nan:
                train_select.append((i, j))
    return train_select


