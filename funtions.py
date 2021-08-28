import streamlit as st

st.cache
import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, PoissonRegressor
import lightgbm as ltb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
from scipy import stats
from matplotlib.gridspec import GridSpec

# Geographic
from functools import partial
from geopy.geocoders import Nominatim
from geopy import distance
from geopy.distance import geodesic

# Evaluate 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from scipy.stats.stats import pearsonr, kurtosis, skew
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Tuning
from sklearn.model_selection import GridSearchCV

def distance__address_cal(ad_1, ad_2):
       
    # Calculate distance between 2 address
    distance = geodesic((geocode(ad_1).latitude, geocode(ad_1).longitude), 
                        (geocode(ad_2).latitude, geocode(ad_2).longitude)).km
    
    return distance
    
    #print(loc_base.address)
    #print((loc_base.latitude, loc_base.longitude))
    #print(loc_base.raw)

def distance__cal(input_1, input_2):
    try:
        distance = geodesic(input_1, input_2).km
    except:
        pass
    return distance

def distance_loop(based_address, facilities, desired_distance):
    distance_lst = []
    in_desired_distance = []
    max
    for i in facilities:
        try:
            distance = distance_cal(based_address, i)
        except:
            distance = 0
        
        distance_lst.append(distance)
    
    # Sort distance_lst ascending
    distance_lst.sort()
    
    # Number of closest location to the based location
    min_5_items_of_list = lst[:,min(5,len(lst))]
    
    # Number of location in the desired_distance
    j = 0
    while distance_lst[j] < desired_distance:
        in_desired_distance.append(lst[j])
        j = j+1
    
    return in_desired_distance, len(in_desired_distancce),\
            min_5_items_of_list, len(min_5_items_of_list)


def get_lat_long(address):
    global count
    try:
        lat = geocode(address).latitude
        long = geocode(address).longitude
        result = "(" + str(lat) + ", " + str(long) + ")"
    except:
        result = ""
    
    count = count + 1
    print(count)
    
    return result

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def count_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_outlier = df_in.loc[(df_in[col_name] <= fence_low) | (df_in[col_name] >= fence_high)]
    outliers = df_outlier.shape[0]
    return outliers

def variance(df_in, col_name):
    lst = list(df_in[col_name])
    
    # Mean of the data
    n = len(lst)
    mean = sum(lst) / n
    
    # Square deviations
    deviations = [(x - mean) ** 2 for x in lst]
    
    # Variance
    variance = sum(deviations) / n
    
    # Standard Deivation
    std_d = variance ** 1/2
    
    return mean, variance, std_d

def count_outlier_by_type(df, col_name):
    # Number of items in each type
    count_by_type = df[['index',col_name]].groupby([col_name]).agg('count').sort_values('index', ascending = False)\
                        .unstack().reset_index()[[col_name, 0]]
    count_by_type = count_by_type.rename(columns={0:'items'})

    # Check outlier in each type
    dist = []
    outs = []
    for i in list(df[col_name].unique()):
        df_to_check = df[df[col_name] == i]
        v = count_outlier(df_to_check, 'avg_price_per_floor')
        dist.append(i)
        outs.append(v)

    dic = {col_name:dist, 'outliers':outs}

    # Join dataframe
    outlier = pd.merge(count_by_type, pd.DataFrame(dic), on=col_name)

    # Create columns %
    outlier['percentage'] = outlier['outliers'].astype(float) / outlier['items'].astype(float) * 100
    outlier['percentage'] = outlier['percentage'].apply(lambda x: str(round(x,1)) + "%")

    # Show result
    tt_outlier = outlier['outliers'].sum()
    tt_base = outlier['items'].sum()

    outlier.sort_values(by='outliers', ascending=False)

    print('Total outliers is:',tt_outlier,'/',tt_base,' or',round(tt_outlier/tt_base*100,2),"%")
    print('Max % of outlier is',outlier['percentage'][0],'in', outlier[col_name][0])
    print('\n',outlier.head(5))

def remove_check_outlier(df, col_name):
    # Check old rows:
    t_1 = df.shape[0]
    
    # Remove outlier in the column
    df = remove_outlier(df, col_name)
    
    # Check new rows:
    t_2 = df.shape[0]
    
    # Print result
    print('Rows before removal:',t_1)
    print('Rows after removal:',t_2)
    print('Number of rows removed:', t_1 - t_2,'accounted for', round((t_2/t_1-1)*100,2),'%')
    
    return df

def filter_by_condition(df, condition):
    t_1 = df.shape[0]
    df = df[condition]
    t_2 = df.shape[0]

    # Print result
    print('Rows before removal:',t_1)
    print('Rows after removal:',t_2)
    print('Number of rows removed:', t_1 - t_2,'accounted for', round((t_2/t_1-1)*100,2),'%')
    
    return df

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df_out = df[indices_to_keep].astype(np.float64).dropna(how='any',axis=0)
    return df_out

def score_estimator(estimator, df_test):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(df_test[input_variable])

    print("MSE: %.3f" %
          mean_squared_error(df_test["price"], y_pred))
    print("MAE: %.3f" %
          mean_absolute_error(df_test["price"], y_pred))

    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(f"WARNING: Estimator yields invalid, non-positive predictions "
              f" for {n_masked} samples out of {n_samples}. These predictions "
              f"are ignored when computing the Poisson deviance.")

    print("mean Poisson deviance: %.3f" %
          mean_poisson_deviance(df_test["price"][mask],
                                y_pred[mask],
                                sample_weight=df_test["price"][mask]))

def visualize(estimator, df_test):
    input_variable = ['alley', 'scaled_log_floor', 
                  'scaled_squared_m2', 'scaled_log_b_room', 
                  'scaled_bhx_1km', 'scaled_vin_1km', 'scaled_market_1km', 'scaled_hospital_1km', 'scaled_school_1km',
                  'scaled_hospital_distance']
    y_pred = estimator.predict(df_test[input_variable])
    
    print('Correlation between y_pred and y_test: %.3f' %
         pearsonr(y_pred, df_test['price'])[0])
    # Visualize
    plt.figure(figsize=(6,6))
    plt.style.use('ggplot')
    plt.scatter(y_pred, df_test['price'])
    plt.xlabel('Model prediction')
    plt.ylabel('True value')
    plt.plot([0,60], [0,60], 'k-', color='r')
    plt.show()

def print_hist(df, col_name, bin, lim):
    plt.style.use('tableau-colorblind10')
    plt.figure(figsize = (6, 4))
    sns.set_context("notebook")
    g = sns.distplot(df[col_name].dropna(), bins = bin)
    g.set(xlim=(0, lim))
    plt.title(str(col_name) + ''' distribution''',x=0.5, y=1, fontsize = 15)
    sns.despine()

# Prepare data
def prepare_data(df, values_col, cat_col):
    df_anova = df[[values_col, cat_col]]
    grps = pd.unique(df_anova[cat_col].values)
    d_data = {grp:df_anova[values_col][df_anova[cat_col] == grp] for grp in grps}
    return d_data

# Print result
def print_statistical_result(p):
    print("p-value for significance is: ", p)
    if p<0.05:
        print("""
    H0: mean of groups are similar
    H1: mean of groups are different

    > reject null hypothesis: mean of groups are different
    """)
    else:
        print("""
    H0: mean of groups are similar
    H1: mean of groups are different

    > accept null hypothesis: mean of groups are different
    """)

def create_ward_tuple(df, district, ward):
    tup = sorted(tuple(df[df.district == district][ward]))
    return tup

def make_prediction(model, min_max, dfOneHot, pr_district, pr_ward, pr_alley, pr_squared_m2,
                    pr_floor, pr_bedroom, pr_bhx, pr_vin, pr_market, pr_hospital, pr_school,
                    pr_hospital_distance):
    # Create dictionary
    administrative_loc = str("S_"+pr_district+" | "+pr_ward)
    raw_input_data = {
        'Distrct': pr_district,
        'Ward': pr_ward,
        'Alley': pr_alley,
        'Squared_m2': pr_squared_m2,
        'Number of floors': int(pr_floor),
        'Number of bedrooms': int(pr_bedroom),
        'Nearby BHX': int(pr_bhx),
        'Nearby Vnmart+': int(pr_vin),
        'Nearby traditional market': int(pr_market),
        'Nearby health care institution': int(pr_hospital),
        'Nearby education institution': int(pr_school),
        'Distance to the nearest healthcare institution': pr_hospital_distance
    }

    input_data = {
        'alley': pr_alley,
        'log_floor': np.log(pr_floor),
        'squared_m2': np.log(pr_squared_m2),
        'log_b_room': np.log(pr_bedroom),
        'bhx_1km': pr_bhx,
        'vin_1km': pr_vin,
        'market_1km': pr_market,
        'hospital_1km': pr_hospital,
        'school_1km': pr_school,
        'hospital_distance': pr_hospital_distance
    }

    test_df = pd.DataFrame(input_data, 
                        columns = ['alley', 'log_floor', 'squared_m2', 'log_b_room', 
                                    'bhx_1km', 'vin_1km', 'market_1km', 'hospital_1km', 'school_1km', 'hospital_distance'], index = [0])

    # Create Columns of administrative loc
    input_var = ['alley', 'scaled_log_floor', 'scaled_squared_m2', 'scaled_log_b_room',
                'scaled_bhx_1km', 'scaled_vin_1km', 'scaled_market_1km', 'scaled_hospital_1km', 'scaled_school_1km', 'scaled_hospital_distance']
    for x in dfOneHot.columns:
        input_var.append(x)
        if x == administrative_loc:
            test_df[x] = float(1)
        else:
            test_df[x] = float(0)
            
    # Scaled value
    cols=['squared_m2', 'log_floor', 'log_b_room', 'bhx_1km', 'vin_1km', 'market_1km', 'hospital_1km', 'school_1km', 'hospital_distance']
    s_cols_name = ['scaled_' + x for x in cols]

    scaled_values = min_max.transform(test_df[cols])
    test_df[s_cols_name] = pd.DataFrame(scaled_values, columns = s_cols_name)
    test_df = test_df[input_var]


    # Predict new value
    st.write('Based on your preferences:\n\n', pd.DataFrame(raw_input_data, index = ['Value']))
    st.write('\nThe price would be: ', round(model.predict(test_df)[0],3))