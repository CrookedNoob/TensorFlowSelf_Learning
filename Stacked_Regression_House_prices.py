# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:38:26 2018

@author: soumyama
"""

#Import necessary libraries
import numpy as np #Linear algebra
import pandas as pd #data processing, CSV file I/O (eg- pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\train.csv')
test = pd.read_csv('C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\test.csv')

#display the first five rows of the train dataset.
train.head(5)

#display the first five rows of the test dataset.
test.head(5)

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


#***********************      DATA PROCESSING      **************************

#OUTLIERS
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
 

sns.distplot(train['SalePrice'], fit=norm)

#Get the fitted parameters used by the function
(mu, sigma) =norm.fit(train['SalePrice'])
print('\n mu= {:.2f} and sigma= {:.2f}\n'.format(mu,sigma))

#Now plot the distribution
plt.legend(['Normal Distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], 
            loc='best')
plt.ylabel('Frequency')
plt.xlabel('SalePrice Distribution')


#Get the QQ plot
fig=plt.figure()
res=stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#SalePrice is skewed. So we need to transform it to amke it more normally distributed

#Log-Transformation of the target variable

#We use the function log1p which applied log(1+x) to all teh elements of the column
train['SalePrice']=np.log1p(train['SalePrice'])


#Check the new distribution
sns.distplot(train['SalePrice'], fit=norm)

#get the fitted parameters used by function
(mu, sigma)=norm.fit(train['SalePrice'])
print('\n mu= {:.2f} and sigma= {:.2f}\n'.format(mu,sigma))

plt.legend(['Normal Distribution ($\mu=$ {:.2f} and $\sigma$ {:.2f})'.format(mu,sigma)], 
            loc='best')
plt.ylabel('Frequency')
plt.xlabel('Transformed SalePrice Distribution')

#QQ-plot
fig=plt.figure()
res=stats.probplot(train['SalePrice'], plot=plt)
plt.show()


#FEATURE ENGINEERING
ntrain=train.shape[0]
ntest=test.shape[0]
y_train=train.SalePrice.values
all_data=pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is :  {}".format(all_data.shape))


#MISSING DATA
all_data_na=(all_data.isnull().sum()/len(all_data))*100
all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]
missing_data=pd.DataFrame({'Missing ratio':all_data_na})
missing_data.head(20)


f, ax =plt.subplots(figsize=(10,10))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of Missing value', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


#DATA CORRELATION

#Correlation map just to see how features are correlated with SalePrice
corrmat=train.corr()
plt.subplots(figsize=(8,8))
sns.heatmap(corrmat, vmax=0.8, square=True)


#IMPUTING MISSING VALUES

#PoolQC : data description says NA means "No Pool". That make sense, given the 
#huge ratio of missing value (+99%) and majority of houses have no Pool at all in general
all_data['PoolQC']=all_data["PoolQC"].fillna("None")

#MiscFeature : data description says NA means "no misc feature"
all_data["MiscFeature"]=all_data["MiscFeature"].fillna("None")

#Alley : data description says NA means "no alley access"
all_data["Alley"]=all_data["Alley"].fillna("None")

#Fence : data description says NA means "no fence"
all_data["Fence"]=all_data["Fence"].fillna("None")

#FireplaceQu : data description says NA means "no fireplace"
all_data["FireplaceQu"]=all_data["FireplaceQu"].fillna("None")

#LotFrontage : Since the area of each street connected to the house property most 
#likely have a similar area to other houses in its neighborhood , we can fill in 
#missing values by the median LotFrontage of the neighborhood.

##Group by neighborhood and fill in missing value by the median LotFrontage of 
#all the neighborhood
all_data["LotFrontage"]=all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))


#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col]=all_data[col].fillna("None")
    
#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 
#(Since No garage = no cars in such garage.)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col]=all_data[col].fillna(0)
    
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : 
#missing values are likely zero for having no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col]=all_data[col].fillna(0)
    
    
#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these 
#categorical basement-related features, NaN means that there is no basement.
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2'):
    all_data[col]=all_data[col].fillna("None")
    
    
#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. 
#We can fill 0 for the area and None for the type. 
all_data["MasVnrType"]=all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"]=all_data["MasVnrArea"].fillna(0)

#MSZoning (The general zoning classification) : 'RL' is by far the most common value. 
#So we can fill in missing values with 'RL'
all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


#Utilities : For this categorical feature all records are "AllPub", except for one 
#"NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this 
#feature won't help in predictive modelling. We can then safely remove it.

all_data=all_data.drop(['Utilities'], axis=1)


#Functional : data description says NA means typical
all_data["Functional"]=all_data["Functional"].fillna("Typ")

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', 
#we can set that for the missing value
all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual: Only one NA value, and same as Electrical, we set 'TA' 
#(which is the most frequent) for the missing value in KitchenQual.
all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. 
#We will just substitute in the most common string
all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


#SaleType : Fill in again with most frequent which is "WD"
all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


#MSSubClass : Na most likely means No building class. We can replace missing values with None
all_data["MSSubClass"]=all_data["MSSubClass"].fillna("None")


#Check the remaining missing values if any
all_data_na=(all_data.isnull().sum()/len(all_data))*100
all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
missing_data=pd.DataFrame({'Missing ratio':all_data_na})
missing_data.head()


    #TRANSFORMING SOME NUMERICAL VARIABLES THAT ARE REALLY CATEGORICAL

##MSSubClass=The building class
all_data['MSSubClass']=all_data['MSSubClass'].apply('str')

#Changing OverallCond into a categorical variable
all_data['OverallCond']=all_data['OverallCond'].apply('str')

#Year and month sold are transformed into categorical features.
all_data['YrSold']=all_data['YrSold'].apply('str')
all_data['MoSold']=all_data['MoSold'].apply('str')


#LABEL ENCODING CATEGORICAL VARIABLES THAT MAY CONTAIN INFORMATION IN THEIR ORDERING SET
from sklearn.preprocessing import LabelEncoder

cols=('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

#Process the columns and apply LabelEncoders to categorical Features
for c in cols:
    lbl=LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c]=lbl.transform(list(all_data[c].values))

#shape
print('Shape all data: {}'.format(all_data.shape))

#ADDING ONE MORE IMPORTANT FEATURE - TOTAL AREA OF BASEMENT AND ALL THE FLOORS
#Adding total square footage
all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']


#SKEWED FEATURES
numeric_feats=all_data.dtypes[all_data.dtypes != "object"].index

#Check the skew of all numeric features
skewed_feats=all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skeweness=pd.DataFrame({'Skew': skewed_feats})
skeweness.head(10)



#BOX COX TRANSFORMATION OF HIGHLY SKEWED FEATURES

#We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x.
#Note that setting λ=0 is equivalent to log1p used above for the target variable.

skeweness=skeweness[abs(skeweness)>0.75]
print("There are {} skewed numerical features to BOX COX transform".format(skeweness.shape[0]))

from scipy.special import boxcox1p
skewed_features=skeweness.index
lam=0.15
for feat in skewed_features:
    all_data[feat]=boxcox1p(all_data[feat], lam)
    
    
#Getting dummy categorical variables
all_data=pd.get_dummies(all_data)
print(all_data.shape)


train = all_data[:ntrain]
test = all_data[ntrain:]




#MODELLING

#Import Libraries
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#import xgboost as xgb
import lightgbm as lgb


#Define a cross validation strategy

#Validation function
n_folds=5

def rmsle_cv(model):
    kf=KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse=np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error",
                                  cv=kf))
    return (rmse)


#BASE MODELS
    
#Lasso Regression
#This model may be very sensitive to outliers. So we need to made it more robust on them.
#For that we use the sklearn's Robustscaler() method on pipeline
lasso=make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

#Elastic Net regression
#made robust to outliers
ENet=make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))


#Kernel Ridge Regression
KRR=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


#Gradient Boosting Regression
#With huber loss that makes it robust to outliers
GBoost= GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,
                                  max_features='sqrt', min_samples_leaf=15, 
                                  min_samples_split=10, loss='huber', random_state=5)


#XGBoost
#model_xgb=xgb.XGBRegressor(colsample_bytree=0.463, gamma=0.0468, learning_rate=0.05,
#                           max_depth=3, min_child_weight=1.7817, n_estimators=2200,
#                           reg_alpha=0.4640, reg_lambda=0.8571,
#                           subsample=0.5213, silent=1,
#                           random_state=7, nthread=-1)


#Light GBM
model_lgb= lgb.LGBMRegressor(objective='regression', num_leaves=5,
                             learning_rate=0.05, n_estimators=720,
                             max_bin=55, bagging_fraction=0.8,
                             bagging_freq=5, feature_fraction=0.2319,
                             feature_fraction_seed=9, baggibg_seed=9,
                             min_data_in_leaf=6, min_sum_hessian_in_leaf=11)



#BASE MODELS SCORES
score =rmsle_cv(lasso)
print("\n Lasso score: {:.4f}  ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(ENet)
print("\n Elastic Net score: {:.4f}  ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(KRR)
print("\n Kernel Ridge score: {:.4f}  ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(GBoost)
print("\n Gradient Boost score: {:.4f}  ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(model_lgb)
print("\n Light GBM score: {:.4f}  ({:.4f})\n".format(score.mean(),score.std()))


#STACKING MODELS

#We begin with this simple approach of averaging base models. We build a new 
#class to extend scikit-learn with our model and also to laverage encapsulation 
#and code reuse (inheritance)

#Averaged base models class

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models=models
    
        #We define clones of the original models to fit the data in
    def fit(self, X,y):
        self.models_=[clone(x) for x in self.models]
        
        
        #Trained cloned base models
        for model in self.models_:
            model.fit(X,y)
        
        return self
    
    #Now we do predictions for cloned models and average them
    def predict(self, X):
        predictions= np.column_stack([
                model.predict(X) for model in self.models_
                ])
        return np.mean(predictions, axis=1)
    
    
    
#Average base model score

#We just average four models here ENet, GBoost, KRR and lasso
averaged_models=AveragingModels(models=(ENet, GBoost, KRR, lasso))

score=rmsle_cv(averaged_models)
print("\nAveraged model base score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#LESS SIMPLE STACKING: ADD A META-MODEL

#STACKING AVERAGED MODELS

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models=base_models
        self.meta_model=meta_model
        self.n_folds=n_folds
        
    #We again fit teh data of clones of the original models
    def fit(self, X,y):
        self.base_models_=[list() for x in self.base_models]
        self.meta_model_=clone(self.meta_model)
        kfold=KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        #Train cloned base models then create out-of-fold predictions
        #that are needed to train the meta model
        out_of_fold_predictions=np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X,y):
                instance=clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred=instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i]=y_pred
                
        #Now train the cloned meta model using teh out_of_fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions,y)
        return self
    
    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    
    def predict(self, X):
        meta_features=np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
    
    

#Stacking Average Model Score
stacked_averaged_models=StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                               meta_model=lasso)

score=rmsle_cv(stacked_averaged_models)
print("\nStacked Averaged model base score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))



#Final Training and Prediction

#Stacked regressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred= stacked_averaged_models.predict(train.values)
stacked_pred=np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred=model_lgb.predict(train)
lgb_pred=np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

#RMSE on the entire training data when averaging
print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred*0.7 + lgb_train_pred*0.3))
    
#Ensemble Prediction
ensemble = stacked_pred*0.7+ lgb_pred*0.3

#Submission
sub=pd.DataFrame()
sub['Id']=test_ID
sub['SalePrice']=ensemble    
sub.to_csv('C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\submission_stacked_models.csv', index=False)
