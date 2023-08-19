##############  Exploratory df Analysis  #######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r"C:\Users\sprav\Desktop\Data Science\Live Project\Sample\Manually Edited Data (1).xlsx")
df.dtypes
bos=pd.DataFrame(df.describe(exclude=[object]))

#List of categorical features
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']
print('Number of categorical variables: ', len(categorical_features))
categorical_features

# list of numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
numerical_features

####################    Continuous   ####################
df_num=df.drop(categorical_features,axis=1)

#First Business Moment

Mean=pd.DataFrame(df_num.mean(),columns=["Mean"]).reset_index()
Median=pd.DataFrame(df_num.median(),columns=["Median"]).reset_index()
m=[]
for i in df_num.columns:
    a = df_num[i].mode().iloc[0]
    m.append(a)
Mode=pd.DataFrame(m,columns=["Mode"]).reset_index()    

#Second Business Moment

Variance=pd.DataFrame(df_num.var(),columns=["Variance"]).reset_index()
Standard=pd.DataFrame(df_num.std(),columns=["Standard_Deviation"]).reset_index()
r=[]
for i in df_num.columns:
    a = max(df_num[i]) - min(df_num[i])
    r.append(a)
Range=pd.DataFrame(m,columns=["Range"]).reset_index()
    
#Third Business Moment
Skew=pd.DataFrame(df_num.skew(),columns=["Skew"]).reset_index()

#Fourth Business Moment
Kurtosis=pd.DataFrame(df_num.kurt(),columns=["Kurtosis"]).reset_index()

eda=pd.concat([Mean["index"],Mean["Mean"],Median["Median"],Mode["Mode"],
               Variance["Variance"],Standard["Standard_Deviation"],Range["Range"],
               Skew["Skew"],Kurtosis["Kurtosis"]],axis=1)

#Preprocessing Techniques

#Outlier Treatment

for i in df_num.columns:
    plt.boxplot(df_num[i])
    plt.title(i)
    plt.show()

#bmi,income and elapsed time since the last complaint needs outlier treatment

#bmi
IQR = df['bmi'].quantile(0.75) - df['bmi'].quantile(0.25)
lower_limit = df['bmi'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['bmi'].quantile(0.75) + (IQR * 1.5)
df['bmi']= pd.DataFrame(np.where(df['bmi'] > upper_limit, upper_limit, 
                                         np.where(df['bmi'] < lower_limit, lower_limit, df['bmi'])))

plt.boxplot(df['bmi'])
plt.title('bmi')

#Income
IQR = df['income'].quantile(0.75) - df['income'].quantile(0.25)
lower_limit = df['income'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['income'].quantile(0.75) + (IQR * 1.5)
df['income']= pd.DataFrame(np.where(df['income'] > upper_limit, upper_limit, 
                                         np.where(df['income'] < lower_limit, lower_limit, df['income'])))

plt.boxplot(df['income'])
plt.title('income')

#Elapsed time since the last complaint
IQR = df['Elapsed_time_since_the_last_complaint'].quantile(0.75) - df['Elapsed_time_since_the_last_complaint'].quantile(0.25)
lower_limit = df['Elapsed_time_since_the_last_complaint'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Elapsed_time_since_the_last_complaint'].quantile(0.75) + (IQR * 1.5)
df['Elapsed_time_since_the_last_complaint']= pd.DataFrame(np.where(df['Elapsed_time_since_the_last_complaint'] > upper_limit, upper_limit, 
                                         np.where(df['Elapsed_time_since_the_last_complaint'] < lower_limit, lower_limit, df['Elapsed_time_since_the_last_complaint'])))
    
plt.boxplot(df['Elapsed_time_since_the_last_complaint'])
plt.title('Elapsed_time_since_the_last_complaint')

#Outstanding_charges
IQR = df['Outstanding_charges'].quantile(0.75) - df['Outstanding_charges'].quantile(0.25)
lower_limit = df['Outstanding_charges'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Outstanding_charges'].quantile(0.75) + (IQR * 1.5)
df['Outstanding_charges']= pd.DataFrame(np.where(df['Outstanding_charges'] > upper_limit, upper_limit, 
                                         np.where(df['Outstanding_charges'] < lower_limit, lower_limit, df['Outstanding_charges'])))

plt.boxplot(df['Outstanding_charges'])
plt.title('Outstanding_charges')

####Transformation

for i in df_num.columns:
    sns.distplot(df_num[i], rug=True, hist=True)
    plt.show() 
    
df_trans=df_num.drop(['Age','Number_of_declarations', 'Number_of_authorizations_',
       'Handling_time_of_authorizations_and_declarations_','Duration_of_current_insurance_contract',
       'Elapsed_time_since_last_contact_moment', 'Product_usage_','Claim_History', 'Renewal_History',
       'Brand_credibility_', 'Claim_After_Renewal','Experience_during_contact_moment', 'Premium_price_'],axis=1)

#df_trans has features which has skewnees value beyond -0.5 & 0.5 permissible range of skewness
#So these features needs to be transformed for better distribution
for i in df_trans.columns:
    sns.distplot(df_trans[i], rug=True, hist=True)
    plt.show() 

for i in df_trans.columns:
    sns.distplot(np.log(df_trans[i]), rug=True, hist=False)
    plt.show()
for i in df_trans.columns:
    sns.distplot(np.sqrt(df_trans[i]), rug=True, hist=False)
    plt.show()
'''for i in df_trans.columns:
    sns.distplot(np.exp(df_num[i]), rug=True, hist=False)
    plt.show()
for i in df_trans.columns:
    sns.distplot(stats.boxcox(df_num[i]), rug=True, hist=False)
    plt.show()    '''

####################    Discrete data   ####################


##### Dummy Variable Creation
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()

for i in categorical_features:
    df[i]= labelEncoder.fit_transform(df[i])

df_cat=df.drop(numerical_features,axis=1)

#Preprocessing Techniques

#Outlier Treatment

for i in df_cat.columns:
    plt.boxplot(df_cat[i])
    plt.title(i)
    plt.show()

####Transformation

for i in df_cat.columns:
    sns.distplot(df_cat[i], rug=True, hist=True)
    plt.show()    
for i in df_cat.columns:
    sns.distplot(np.log(df_cat[i]), rug=True, hist=False)
    plt.show()
for i in df_cat.columns:
    sns.distplot(np.sqrt(df_cat[i]), rug=True, hist=False)
    plt.show()
for i in df_cat.columns:
    sns.distplot(np.exp(df_cat[i]), rug=True, hist=True)
    plt.show()
'''for i in df_cat.columns:
    sns.distplot(stats.boxcox(df_cat[i]), rug=True, hist=False)
    plt.show()  '''
    
df.to_csv(r'C:\Users\sprav\Desktop\Data Science\Live Project\360\Final_Dataset.csv', index=False)
