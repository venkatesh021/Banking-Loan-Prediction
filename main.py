'''Importing all the required files for this project'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
#accessing the training and testing data for operations
train_df=pd.read_csv("Train.csv")
test_df=pd.read_csv("Test.csv")
print('Sample view of training data:\n',train_df.head())
print('Sample view of testing data:\n',test_df.sample(5))
print('Total no of rows and columns in train data are:',train_df.shape)
print('Total no of rows and columns in train data are:',test_df.shape)
# Checking columns for Null values
def null_cols(d,columns):
  na = d[columns].isna().sum().sort_values(ascending=False)
  return na[na > 0].index.to_list()
# Replacing all null values
def replace_null(X,cols):
  na_cols = null_cols(X,cols)
  print('Number Null columns presented before filling',len(na_cols))
  if len(na_cols) > 0:
     # Replacing null values using ffill and bfill options
    X[na_cols]= X[na_cols].fillna(method='ffill')
    X[na_cols]= X[na_cols].fillna(method='bfill')
    na_cols = null_cols(X,cols)
    print('Number of Null columns present after filling',len(na_cols))
target = 'Col2'
num_cols = train_df.select_dtypes(include="number").columns
cat_cols = list(set(train_df.columns) - set(num_cols))
corr = train_df.corr()#finding correlation for training data using corr() function
k= 50
'''Assining Only maximum valued columns to the cols variable which are contributing more for 
data analysis'''
cols = corr.nlargest(k,target)[target].index
print(cols)
cm = train_df.loc[:,cols].corr()
f,ax = plt.subplots(figsize = (16,12))
col_x = list(set(cols) - set([target]))
X,Y = train_df.loc[:,col_x] ,train_df.loc[:,target]
replace_null(X,col_x)
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=False,cmap='viridis',linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
plt.show()#ploting a heat map for knowing which columns has highest impact on required output column
def generate_accuracy_and_heatmap(model, x, y):
    ac = accuracy_score(y,model.predict(x))
    f_score = f1_score(y,model.predict(x))
    print('Accuracy is: ', ac)
    print('F1 score is: ', f_score)
    print ("\n")
    pass
print('Before OverSampling, the shape of train_X: {}'.format(X.shape))
print('Before OverSampling, the shape of train_y: {} \n'.format(Y.shape))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X, Y)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
#creating a variable for finding regression values
clf_lr = LogisticRegression()
lr_baseline_model = clf_lr.fit(X, Y)
generate_accuracy_and_heatmap(lr_baseline_model, X, Y)
print(pd.crosstab(clf_lr.predict(X),Y,rownames=['Predicted'], colnames=['Actual']))
# fit model no training data
model = XGBClassifier(learning_rate=1,n_jobs=4)
model.fit(X, Y)
y_pred = model.predict(X)
predictions = [round(value) for value in Y]
pd.crosstab(y_pred,Y,rownames=['Predicted'], colnames=['Actual'])
generate_accuracy_and_heatmap(model, X, Y)
weights = np.linspace(0.05, 0.95, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={'class_weight': [{0: x, 1: 1.0-x} for x in weights]},
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X, Y)
print("Best parameters : %s" % grid_result.best_params_)
data_out = pd.DataFrame({'score': grid_result.cv_results_['mean_test_score'],'weight': weights })
data_out.plot(x='weight')#ploting a line graph to know the weights of the column values
plt.show()
clf = LogisticRegression(**grid_result.best_params_).fit(X, Y)
print(clf)
y1 = clf.predict(X)
# Constructing a Confusion matrix
print(pd.crosstab(y1,Y,rownames=['Predicted'], colnames=['Actual']))
test_df.loc[:,col_x] = test_df.loc[:,col_x].fillna(method='ffill')
test_df.loc[:,col_x].isna().sum().sort_values(ascending=False)
'''Creating a sample csv file for finding our required target column. For that i have created a 
Sample_submission column which don't contains any values just prepared for sample purpose .'''
sample = pd.read_csv("Sample_submission.csv")
y_test = model.predict(test_df.loc[:,col_x])
print(sum(y_test))
sample.drop(index=sample.index,inplace=True)
sample['Col1'] = test_df.iloc[:,0]
sample['Col2'] = y_test
sample.to_csv('out.csv',index=False)
sample=sample.drop(['column1','column2'],axis=1)
print("Sample data of required output:\n",sample.sample(20))
print("Total no of columns and rows present in targeted output:\n",sample.shape)
