#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITA Model Development 

@author: mobashirshandhi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats




### function to get summary statistics of deviation metrics from time series wearable data baseline to detection period 




def get_delta_summary_stat(wearable_daily_df, baseline_start_date,baseline_end_date,detection_start_date,detection_end_date):

###  Example baseline and detection period start and end date    
###     baseline_start_date=-60
###     baseline_end_date=-22
###     detection_start_date=-5
###     detection_end_date=-1

    ## get separate dataframe for baseline and detection period
    wearable_daily_df_bl=wearable_daily_df.loc[(wearable_daily_df['days_from_test'] >= baseline_start_date) & (wearable_daily_df['days_from_test'] <= baseline_end_date)]
    wearable_daily_df_detection=wearable_daily_df.loc[(wearable_daily_df['days_from_test'] >= detection_start_date) & (wearable_daily_df['days_from_test'] <= detection_end_date)]
    wearable_daily_df_detection.reset_index(inplace=True,drop=True)
    wearable_daily_df_bl.reset_index(inplace=True,drop=True)



    ## get subjectwise summary (median, mean, std, iqr) for baseline period
    wearable_daily_df_bl_subject_wise_summary=wearable_daily_df_bl.groupby(['participant_id'],as_index=True).agg(
        baseline_median = ('Value', np.median),
        baseline_mean = ('Value', np.mean),
        baseline_std = ('Value', np.std),
        baseline_iqr = ('Value', stats.iqr)
    )


    # get deviation metrics (deviation from baseline) for each day in the detection period
    
    ## adding bl summary value with the detection df
    wearable_daily_df_detection = wearable_daily_df_detection.join(wearable_daily_df_bl_subject_wise_summary, on="participant_id")
    ## calculating deviation metrics
    wearable_daily_df_detection['Delta']=wearable_daily_df_detection['Value']-wearable_daily_df_detection['baseline_median']
    wearable_daily_df_detection['Delta_Normalized']=wearable_daily_df_detection['Delta']/wearable_daily_df_detection['baseline_median']
    wearable_daily_df_detection['Delta_standardized']=wearable_daily_df_detection['Delta']/wearable_daily_df_detection['baseline_iqr']
    wearable_daily_df_detection['Z_score']=(wearable_daily_df_detection['Value']-wearable_daily_df_detection['baseline_mean'])/wearable_daily_df_detection['baseline_std']


    ## get summary statistics for each deviation metrics
    wearable_daily_df_detection_summary_stat = wearable_daily_df_detection.groupby('participant_id').agg(
            delta_avg = ('Delta', np.mean),
            delta_max = ('Delta', max),
            delta_min = ('Delta', min),
            delta_range = ('Delta', np.ptp),
            delta_median = ('Delta', np.median),
            normalized_delta_avg = ('Delta_Normalized', np.mean),
            normalized_delta_max = ('Delta_Normalized', max),
            normalized_delta_min = ('Delta_Normalized', min),
            normalized_delta_range = ('Delta_Normalized', np.ptp),
            normalized_delta_median = ('Delta_Normalized', np.median),
            standardized_delta_avg = ('Delta_standardized', np.mean),
            standardized_delta_max = ('Delta_standardized', max),
            standardized_delta_min = ('Delta_standardized', min),
            standardized_delta_range = ('Delta_standardized', np.ptp),
            standardized_delta_median = ('Delta_standardized', np.median),
            zscore_delta_avg = ('Z_score', np.mean),
            zscore_delta_max = ('Z_score', max),
            zscore_delta_min = ('Z_score', min),
            zscore_delta_range = ('Z_score', np.ptp),
            zscore_delta_median = ('Z_score', np.median),
            covid_status=('covid_status', np.mean)
            )

    
    return wearable_daily_df_detection_summary_stat


### Function Run two sample test on features based on a grouping variable (covid_status) and return p-value plus summary metrics
def parametric_two_sample_ttest(df):
    column_list = [x for x in df.columns if x != 'covid_status']
    # create an empty dictionary
    t_test_results = {}
    # loop over column_list and execute code 
    for column in column_list:
        group1 = df.where(df.covid_status== 1).dropna()[column] # for covid positive group
        group2 = df.where(df.covid_status== 0).dropna()[column] # for covid positive group
        # add the output to the dictionary 
        t_test_results[column] = stats.ttest_ind(group1,group2)
        


    results_df = pd.DataFrame.from_dict(t_test_results,orient='Index')
    results_df.columns = ['statistic','pvalue']
    results_df.drop(columns=['statistic'],inplace=True)


    return results_df



### Function to train a model on a dataset with nested cross validation with outer loop as model selection and inner loop for hyperparameter tuning
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_val_predict,StratifiedKFold
from sklearn.preprocessing import StandardScaler

## define nested cross validation
def nested_cross_validate_predict_proba(X, y, model, param_grid):
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(5)
    cv_outer=StratifiedKFold(10)
    

    # set up the inner CV loop
    clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='recall', cv=cv_inner,n_jobs=-1)    
    # set up outer CV loop
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('model', clf)])
    # execute the nested cross-validation
    probabilities = cross_val_predict(pipe, X, y,cv=cv_outer,n_jobs = -1,verbose = True,method='predict_proba')
    
    return probabilities





### Import Preprocessed wearable data with daily RHR and step count data
# Column Names:participant_id, covid_status, rhr, step_count, days_from_test
# covid_status is a integer value: covid positive (1)/ covid negative (0)
wearable_df = pd.read_csv('wearable_digital_biomarkers.csv', index_col = False)
#defining baseline and detection period
baseline_start_date=-60
baseline_end_date=-22
detection_start_date=-5
detection_end_date=-1

### Extract Features from time series wearable data
#get features for RHR
wearable_rhr_feature_df=get_delta_summary_stat(wearable_df.rename(columns = {'rhr': 'Value'}),baseline_start_date, baseline_end_date, detection_start_date,detection_end_date) # 3 days of Detecion Period
wearable_rhr_feature_df=wearable_rhr_feature_df.add_prefix('rhr_')

#get features for steps
wearable_step_feature_df=get_delta_summary_stat(wearable_df.rename(columns = {'step_count': 'Value'}),baseline_start_date, baseline_end_date, detection_start_date,detection_end_date) 
wearable_step_feature_df=wearable_step_feature_df.add_prefix('step_')


wearable_feature_df=wearable_step_feature_df.join(wearable_rhr_feature_df,on="participant_id")
#get covid_status from both rhr and step data
wearable_feature_df['covid_status']=wearable_feature_df[['rhr_covid_status','step_covid_status']].max(axis=1)   


### split the dataset into training and test set with a 80-20 ratio  
### keep the ratio of the covid positive and negative participants the same in both training and test set
from sklearn.model_selection import train_test_split

participants_list=wearable_feature_df.index.tolist()

covid_status=wearable_feature_df['covid_status']
participants_list_train, participants_list_test, covid_status_train,covid_status_test=train_test_split(participants_list, covid_status, test_size=0.2, random_state=42, stratify=covid_status)
wearable_feature_train_df = wearable_feature_df[wearable_feature_df['participant_id'].isin(participants_list_train)]
wearable_feature_test_df = wearable_feature_df[wearable_feature_df['participant_id'].isin(participants_list_test)]



### feature selection using statistical analysis on the training set
### run two sample ttest on the training set and only use the features that statistically significant between covid positive and negative groups
wearable_feature_train_df.reset_index(inplace=True)
stat_results=parametric_two_sample_ttest(wearable_feature_train_df.drop(columns=['participant_id', 'rhr_covid_status','step_covid_status'])) # only pass the columns with features and covid_Status
# adjust p-values using Benjamini-Hochberg procedure for multiple hypotheses testing
from statsmodels.stats.multitest import multipletests

adjusted_p_value=multipletests(stat_results['pvalue'], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
stat_results['pvalue']=adjusted_p_value[1]
#get the list of significant features with p-value < 0.05
stat_results.reset_index(inplace=True)
significant_features=stat_results[stat_results['pvalue']<0.05]['index']



### Run ITA Classification Model
target_and_grouping_features=['participant_id','covid_status']
wearable_features=list(significant_features)
features_to_extract=target_and_grouping_features+wearable_features

# extract only the relevant columns from the dataframe
train_df=wearable_feature_train_df[features_to_extract].copy()

# defining target variable and estimating variables
X = train_df[wearable_features].copy() # Features
y = train_df.covid_status # Target variable


### Define model for the imbalanced dataset
classifier=LogisticRegression(class_weight='balanced', max_iter=1000)
# define parameter grid
param_grid={"C":[0.001, 0.01, 0.1, 1, 10, 100],
           "penalty":['l1','l2'],
           "solver":['liblinear','lbfgs']}

# get problity of each individual of getting tested positive for covid-19
y_prob=nested_cross_validate_predict_proba(X, y, classifier, param_grid)


#creating a dataframe with the true value and probability
ranking_df=pd.DataFrame(list(zip(y,y_prob)),columns = ['True_Class', 'Predicted Probability'])

# create a ranked list of participants with higher likelihood getting tested positive for covid-19
ranking_df.sort_values(by=['Predicted Probability'],inplace=True,ascending=False,ignore_index=True)
ranking_df.reset_index(inplace=True)



## setting up a scenario of limited diagnostic setting with X% available test for overall population
percentage_of_rows=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 10-100% of the population
data=np.zeros((len(percentage_of_rows), 3))
for i in range(0,len(percentage_of_rows)):
    number_of_rows=round(len(ranking_df.index)*percentage_of_rows[i]) 
    ranking_df_subset=ranking_df.head(number_of_rows)
    data[i,0]=round(len(ranking_df.index)*percentage_of_rows[i]) # number of data points
    data[i,1]=ranking_df_subset['True_Class'].value_counts()[1] # number of covid positive
    data[i,2]=ranking_df_subset['True_Class'].value_counts()[0] # number of covid negative

    
    
    

ranking_df_range=pd.DataFrame(data=data, columns=['number_of_available_tests', 'number_of_covid_positive', 'number_of_covid_negative'],dtype=None, copy=None)
ranking_df_range['overall_number_of_covid_positive']=ranking_df['True_Class'].value_counts()[1]
ranking_df_range['overall_percentage_of_covid_positive']=100*(ranking_df_range['overall_number_of_covid_positive']/len(ranking_df.index))
ranking_df_range['positivity_rate']=100*(ranking_df_range['number_of_covid_positive']/ranking_df_range['number_of_tests'])


# create a plot with resultant positivity rate from ITA-identified subpopulation
fig = plt.figure(figsize=(6, 6))
plt.rcParams.update({'font.size': 12})
ax = fig.add_subplot(1,1,1)


sns.lineplot(data=ranking_df_range,x='number_of_tests',y='positivity_rate',legend=False,color='blue',ax=ax)
sns.lineplot(data=ranking_df_range,x='number_of_tests',y='overall_percentage_of_covid_positive',legend=False,color='red',linestyle='--',ax=ax)

ax.set_xlabel('Number of Available Diagnostic Tests')
ax.set_ylabel('Positivity Rate')
ax.set_ylim(0, 100)

plt.title('Intelligent Testing Allocation Subpopulation') 
plt.savefig('Plot of Testing Allocation.jpg',format='jpg', dpi = 400)

plt.show()   

