
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 4 - Building Good Training Sets – Data Preprocessing

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).


# ### Overview

# - [Dealing with missing data](#Dealing-with-missing-data)
#   - [Eliminating samples or features with missing values](#Eliminating-samples-or-features-with-missing-values)
#   - [Imputing missing values](#Imputing-missing-values)
#   - [Understanding the scikit-learn estimator API](#Understanding-the-scikit-learn-estimator-API)
# - [Handling categorical data](#Handling-categorical-data)
#   - [Mapping ordinal features](#Mapping-ordinal-features)
#   - [Encoding class labels](#Encoding-class-labels)
#   - [Performing one-hot encoding on nominal features](#Performing-one-hot-encoding-on-nominal-features)
# - [Partitioning a dataset into separate training and test sets](#Partitioning-a-dataset-into-seperate-training-and-test-sets)
# - [Bringing features onto the same scale](#Bringing-features-onto-the-same-scale)
# - [Selecting meaningful features](#Selecting-meaningful-features)
#   - [Sparse solutions with L1 regularization](#Sparse-solutions-with-L1-regularization)
#   - [Sequential feature selection algorithms](#Sequential-feature-selection-algorithms)
# - [Assessing feature importance with Random Forests](#Assessing-feature-importance-with-Random-Forests)
# - [Summary](#Summary)

# <br>
# <br>

# In[2]:


# # Dealing with missing data

# In[3]:

import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
df


# In[4]:

df.isnull().sum()


# In[5]:

# access the underlying NumPy array
# via the `values` attribute
df.values


# <br>
# <br>

# ## Eliminating samples or features with missing values

# In[6]:

# remove rows that contain missing values

df.dropna(axis=0)


# In[7]:

# remove columns that contain missing values

df.dropna(axis=1)


# In[8]:

# only drop rows where all columns are NaN

df.dropna(how='all')  


# In[9]:

# drop rows that have not at least 4 non-NaN values

df.dropna(thresh=4)


# In[10]:

# only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=['C'])


# <br>
# <br>

# ## Imputing missing values

# In[11]:

# again: our original array
df.values


# In[12]:

# impute missing values via the column mean

from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data


# <br>
# <br>

# ## Understanding the scikit-learn estimator API



# <br>
# <br>

# # Handling categorical data

# In[15]:

import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
df


# <br>
# <br>

# ## Mapping ordinal features

# In[16]:

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df


# In[17]:

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)


# <br>
# <br>

# ## Encoding class labels

# In[18]:

import numpy as np

# create a mapping dict
# to convert class labels from strings to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping


# In[19]:

# to convert class labels from strings to integers
df['classlabel'] = df['classlabel'].map(class_mapping)
df


# In[20]:

# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df


# In[21]:

from sklearn.preprocessing import LabelEncoder

# Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y


# In[22]:

# reverse mapping
class_le.inverse_transform(y)


# <br>
# <br>

# ## Performing one-hot encoding on nominal features

# In[23]:

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X


# In[24]:

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()


# In[25]:

# return dense array so that we can skip
# the toarray step

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
ohe.fit_transform(X)


# In[26]:

# one-hot encoding via pandas

pd.get_dummies(df[['price', 'color', 'size']])


# <br>
# <br>

# # Partitioning a dataset into seperate training and test sets

# In[27]:

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()


# In[28]:

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)


# <br>
# <br>

# # Bringing features onto the same scale

# In[29]:

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# In[30]:

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# A visual example:

# In[31]:

ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))


# <br>
# <br>

# # Selecting meaningful features

# ...

# ## Sparse solutions with L1-regularization

# In[32]:



# In[35]:

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# In[36]:

lr.intercept_


# In[37]:

np.set_printoptions(8)


# In[38]:

lr.coef_[lr.coef_!=0].shape


# In[39]:

lr.coef_


# In[40]:

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
plt.show()


# <br>
# <br>

# ## Sequential feature selection algorithms

# In[41]:

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# In[42]:

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()


# In[43]:

k4 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k4])


# In[44]:

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))


# In[45]:

knn.fit(X_train_std[:, k4], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k4], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k4], y_test))


# <br>
# <br>

# # Assessing feature importance with Random Forests

# In[46]:

from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.savefig('images/04_09.png', dpi=300)
plt.show()


# In[47]:

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of samples that meet this criterion:', 
      X_selected.shape[0])


# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):

# In[48]:

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


# <br>
# <br>

# # Summary

# ...
