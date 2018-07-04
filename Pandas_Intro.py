# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:37:14 2018

@author: ASUS
"""


import pandas as pd
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

cities= pd.DataFrame({'City name': city_names, 'Population':population})
print(type(cities['City name']))
cities['City name']

population/1000


import numpy  as np

np.log(population)

cities['Area in sq miles']=pd.Series([46.87, 176.53,97.92])

cities['Population Density']= cities['Population']/cities['Area in sq miles']

cities

cities['Is wide and has saint name']= (cities['Area in sq miles']>50)&cities['City name'].apply(lambda name: name.startswith('San'))
cities

cities['City name'].index

cities.reindex([2,1,0])


cities.reindex(np.random.permutation(cities.index))


cities.reindex([5,3,0])
