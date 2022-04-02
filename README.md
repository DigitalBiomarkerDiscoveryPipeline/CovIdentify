## CovIdentify Study
CovIdentify study was launched in April 2020 to integrate commercial wearable device data and electronic symptom surveys to calculate an individual’s real-time risk of being infected with COVID-19.

![COVIDENTIFY_medium](https://user-images.githubusercontent.com/65799761/161335603-703259d3-004e-4755-a115-ea1b4aa7ce8d.png)

## Intelligent Testing Allocation (ITA) Algorithm:
To determine who to test in settings where there are a limited number of diagnostic tests available (i.e., limited testing capacity), we explored whether information from wearables could help rank individuals by their likelihood of a current COVID-19 infection. To achieve this, we developed an Intelligent Testing Allocation (ITA) model which leverages differences in digital biomarkers (e.g., daily resting heart rate (RHR), step count, sleep duration) from wearable devices (smart watch: Fitbit, Garmin, Apple Watch) to distinguish individuals who are likely positive or negative for COVID-19 in order to improve current methods of diagnostic test allocation and increase testing positivity rates. 

# Preprocessing: 
For participants with high frequency (second- or minute- level, depending on device types) wearable data, we calculated daily RHR and step count. For the remaining participants (where high frequency data were unavailable), we used device-reported daily values for HR and step count.

Given the use of datasets with different device types, a consistent RHR definition was used in order to harmonize the cohorts with high frequency wearable data. We calculated the daily RHR digital biomarker by aggregating the high frequency heart rate data points available between midnight and 7 AM, when there were no steps recorded. Step count was calculated by summing all recorded step values during a 24 hour period in order to produce a daily step count digital biomarker.

# Feature Engineering and Extraction: 
We first calculated four deviation metrics (delta, delta_normalized, delta_standardized, and z_score), which capture the deviation in digital biomarkers from participants’ baseline during the detection phase. Following the deviation metrics calculation, we calculated summary statistics (mean, median, maximum, minimum, and range) of these four deviation metrics which we refer as to features for the ITA model deveopment. We used the function get_delta_summary_stat in the ita_model_development.py to extract the features. 


# Feature Selection:
We only used training set data for feature exploration and selection. We performed statistical analysis on the features from the training dataset to see which features are statistically different between the two groups. We utilized multiple hypothesis testing with Benjamini-Hochberg adjusted p-values for this statistical analysis. Following the statistical analysis, we only utilized the statistically significant features (p-value < 0.05)  for the development of the ITA model. 

# ITA Model Development:
Following feature extraction, we developed predictive models to classify COVID-19 positive and negative participants in the training dataset  using nested cross-validation (CV) and later validated the models the independent test datasets. We predicted the proability of each individual to be tested positive for COVID-19 using this classification model.

# Improvement in positivity rate for COVID-19 diagnostic testing using the ITA method:
We next evaluated how the ITA model can improve the current standard of practice for COVID-19 infection surveillance. We ranked the individual using our ITA model generated probaility of testing positive for COVID-19. Furthermore, we created a hypothetical scenario of limited resource settings, e.g., we can only test X% (10-30%) of overall population. We chose a subpopulation based on the ITA-generated rank (top X% from the ranked list) based on available diagnostic tests and calculated the positivity rate of the subpopulation compared to the random-selection based positivity rate (percentage of covid positive group in the overall population).

![Line Plot of Testing Allocation using ITA](https://user-images.githubusercontent.com/65799761/161372937-5d87474f-ed74-4003-a20d-07f45aa92282.jpg)




