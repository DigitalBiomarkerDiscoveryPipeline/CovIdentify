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
import os




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


### Import Preprocessed wearable data with daily RHR and step count data with participant_id, covid_status (1/0), rhr, step_count, days_from_test
wearable_df = pd.read_csv('XXXX.csv', index_col = False)
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






