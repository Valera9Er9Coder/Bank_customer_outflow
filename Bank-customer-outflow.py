#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import accuracy_score , classification_report , ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[2]:


Sumple_submission = pd.read_csv('C:/Users/Hp/Desktop/Отток Клиентов/Sumple_submission.csv')
Sumple_submission.head()


# In[3]:


Sumple_submission.info()


# In[4]:


print(f'Средний возвраст:' , Sumple_submission['Age'].mean())
print(f'Средний кредитный рейтинг:' , Sumple_submission['CreditScore'].mean())


# In[5]:


numeric_data = Sumple_submission.select_dtypes(include = ['float64','int64'])
plt.figure(figsize = (12,12))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot = True ,cmap = 'coolwarm' )
plt.title('Матрица корреляций')
plt.show()


# In[6]:


plt.figure(figsize = (6,6))
sns.histplot(Sumple_submission['CreditScore'],kde = True , bins = 50 , color = 'blue')
plt.title('Распределение кредитных рейтингов')
plt.show()


# In[7]:


def credit_lavel(scores):
    if scores < 600:
        return 'Low'
    elif scores  <800:
        return 'Medium'
    else:
        return 'Hight'

Sumple_submission['Credit_level'] = Sumple_submission['CreditScore'].apply(credit_lavel)
print(f'Новый признак:' , Sumple_submission['Credit_level'])


# In[8]:


Sumple_submission.drop(['CustomerId','Surname','id'],axis = 1 , inplace = True)


# In[9]:


Sumple_submission.head()


# In[10]:


Sumple_submission = pd.get_dummies(Sumple_submission, columns = ['Geography','Gender'],drop_first = True)
Sumple_submission


# In[11]:


Sumple_submission.columns


# In[12]:


# Список столбцов для масштабирования
numerical_features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']

# Инициализируем StandardScaler
scaler = StandardScaler()

# Применяем масштабирование только к существующим столбцам
Sumple_submission[numerical_features] = scaler.fit_transform(Sumple_submission[numerical_features])

# Выводим результат
print(Sumple_submission.head())


# In[13]:


X = Sumple_submission.drop('Exited' , axis =1)
y = Sumple_submission['Exited']


# In[14]:


# Выделение категориальных колонок
categorical_columns = X.select_dtypes(include=['object']).columns

# Создание OneHotEncoder
one_hot_encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'  # Сохранение других столбцов без изменений
)

# Преобразование данных
X = one_hot_encoder.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model_knn = KNeighborsClassifier(n_neighbors = 5)
model_knn.fit(X_train , y_train)


# In[16]:


knn_pred = model_knn.predict(X_test)
print(accuracy_score(y_test , knn_pred))
print(classification_report(y_test , knn_pred))


# In[17]:


rf_model = RandomForestClassifier(n_estimators = 200 , random_state = 42)
rf_model.fit(X_train , y_train)


# In[18]:


rf_pred = rf_model.predict(X_test)
print(accuracy_score(y_test , rf_pred))
print(classification_report(y_test , rf_pred))


# In[19]:


model_gbc = GradientBoostingClassifier(random_state = 42)
model_gbc.fit(X_train , y_train)


# In[20]:


gbc_pred = model_gbc.predict(X_test)
print(accuracy_score(y_test , gbc_pred))
print(classification_report(y_test , gbc_pred))


# In[21]:


knn_scores = cross_val_score(model_knn,X,y,cv = 5)
print('KNN Cross_Vallidation Scores:' , knn_scores)
print('KNN Mean CV Accuracy:',knn_scores.mean())


# In[ ]:


rf_scores = cross_val_score(rf_model,X,y,cv=5)
print('Random Forest Cross_Vallidation Scores:',rf_scores)
print('Random Foredt Mean CVAccuracy:',rf_scores.mean())


# In[ ]:


ConfusionMatrixDisplay.from_estimator(model_knn,X_test,y_test)
plt.title('Confussion Matrix for KNN')
plt.show()


# In[ ]:


ConfusionMatrixDisplay.from_estimator(rf_model , X_test , y_test)
plt.title('Confussin Matrix for Random Forest')
plt.show()


# In[ ]:


ConfusionMatrixDisplay.from_estimator(model_gbc , X_test , y_test)
plt.title('Confussion Matrix for Gradient Boosting')
plt.show()


# In[ ]:


Sumple_submission['Exited']= y


# In[ ]:


Sumple_submission.to_csv('C:/Users/Hp/Desktop/Отток Клиентов/processed_customer_churn.csv', index = False)


# In[ ]:




