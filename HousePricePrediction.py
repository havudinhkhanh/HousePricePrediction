import streamlit as st
import joblib
#import funtions

#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

st.cache
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime
#import lazypredict
#from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression, PoissonRegressor
#import lightgbm as ltb
#from sklearn.svm import SVR
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
from scipy import stats
from matplotlib.gridspec import GridSpec

# Geographic
#from functools import partial
#from geopy.geocoders import Nominatim
#from geopy import distance
#from geopy.distance import geodesic

# Evaluate 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from scipy.stats.stats import pearsonr, kurtosis, skew
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Tuning
from sklearn.model_selection import GridSearchCV

# Load model
filename = 'poisson_glm.sav'
poisson_modl = joblib.load(filename)

# Load dataset
new_df = pd.read_csv('new_df.csv', sep = '\t', header = 0)
di_wa_df = new_df[['district', 'ward']]
di_wa_df = di_wa_df.dropna().drop_duplicates()

# Data Cleaning
##-----------------------------------------------------------##
### Bách Hóa Xanh--------------------------------###
#df['bhx_1km'] = df['min_bhx'].apply(lambda x: x.split(' | ')[0])
#df['bhx_1km'] = df['bhx_1km'].astype('float')

#df['bhx_distance'] = df['min_bhx'].apply(lambda x: x.split(' | ')[1])
#df['bhx_distance'] = df['bhx_distance'].astype('float')

### Vinmart+--------------------------------------###
#df['vin_1km'] = df['min_vin'].apply(lambda x: x.split(' | ')[0])
#df['vin_1km'] = df['vin_1km'].astype('float')

#df['vin_distance'] = df['min_vin'].apply(lambda x: x.split(' | ')[1])
#df['vin_distance'] = df['vin_distance'].astype('float')

### Tradditional Market----------------------------###
#df['market_1km'] = df['min_market'].apply(lambda x: x.split(' | ')[0])
#df['market_1km'] = df['market_1km'].astype('float')

#df['market_distance'] = df['min_market'].apply(lambda x: x.split(' | ')[1])
#df['market_distance'] = df['market_distance'].astype('float')

### Hospital----------------------------------------###
#df['hospital_1km'] = df['min_hospital'].apply(lambda x: x.split(' | ')[0])
#df['hospital_1km'] = df['hospital_1km'].astype('float')

#df['hospital_distance'] = df['min_hospital'].apply(lambda x: x.split(' | ')[1])
#df['hospital_distance'] = df['hospital_distance'].astype('float')

### Education---------------------------------------###
#df['school_1km'] = df['min_school'].apply(lambda x: x.split(' | ')[0])
#df['school_1km'] = df['school_1km'].astype('float')

#df['school_distance'] = df['min_school'].apply(lambda x: x.split(' | ')[1])
#df['school_distance'] = df['school_distance'].astype('float')

### Drop unneccesary cols----------------------------###
#df = df.drop(columns=['min_vin', 'min_bhx', 'min_market', 'min_school', 'min_hospital'])

## Missing value----------------------------------------------------###
#df = funtions.filter_by_condition(df, df.floor != 0)
#df = funtions.filter_by_condition(df, df.floor.isnull()==False)
#df = funtions.filter_by_condition(df, df.b_room != 0)
#df.b_room = df.b_room.astype('float')
#df = funtions.filter_by_condition(df, df.squared_m2 != 0)
#df = funtions.filter_by_condition(df, df.squared_m2.isnull()==False)
#df = df.drop(columns=['wc_room', 'id', 'text', 'description', 'legal', 'direction'])

## Outliers----------------------------------------------------------###
#df = funtions.remove_check_outlier(df, 'squared_m2')
#df.reset_index(inplace = True)

### Create new features----------------------------------------------###
#df['bin_floors'] = df.floor.apply(lambda x: 5 if x >= 5 else x) 
#df['bin_bedroom'] = df.b_room.apply(lambda x: 10 if x >= 10 else x)
#df['da'] = df.district + " | " + df.alley.apply(lambda x: str(x))
#df['dw'] = df.district + " | " + df.ward
#df['dwa'] = df.district + " | " + df.ward + " | " + df.alley.apply(lambda x: str(x))
#df['dwafb'] = df['dwa'] + " | " + str(df['bin_floors']) + str(df['bin_bedroom'])
#df['avg_price_per_floor'] = df.avg_price / df.floor
#df['avg_price_per_floor'] = df['avg_price_per_floor'].astype(float)

### Remove outliers---------------------------------------------------###
#new_df = pd.DataFrame(columns = list(df.columns))
#for i in list(df.dwafb.unique()):
#    df_in = df[df.dwafb == i]
#    df_result = funtions.remove_outlier(df_in, 'avg_price_per_floor')
#    new_df = new_df.append(df_result)
#new_df.reset_index(inplace = True)

## Standardize features-----------------------------------------------###
### Log scale---------------------------------------------------------###
#new_df['floor'] = new_df['floor'].astype(float)
#new_df['log_floor'] = np.log(new_df.floor)
#new_df['b_room'] = new_df['b_room'].astype(float)
#new_df['log_b_room'] = np.log(new_df.b_room)

### Min Max Scaler----------------------------------------------------###
#new_df = new_df.drop(columns=['level_0'])
#new_df = new_df.reset_index()
#new_df = pd.read_csv('new_df.csv',header=0, sep = '\t')
min_max = preprocessing.MinMaxScaler()
columns=['squared_m2', 'log_floor', 'log_b_room', 
         'bhx_1km', 'vin_1km', 'market_1km', 'hospital_1km', 'school_1km', 'hospital_distance']
scaled_columns = ['scaled_' + x for x in columns]
new_df_after_min_max_scaler = min_max.fit_transform(new_df[columns])
new_df[scaled_columns] = pd.DataFrame(new_df_after_min_max_scaler, columns = scaled_columns)

### OneHotEncoder------------------------------------------------------###
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(new_df[['dw']])
onehot_array = one_hot_encoder.transform(new_df[['dw']]).toarray()
dfOneHot = pd.DataFrame(onehot_array, columns = ['S_'+i for i in one_hot_encoder.categories_[0]])
new_df = new_df.reset_index()
#new_df = new_df.drop(columns=['index']).reset_index()
new_df_onehot = pd.concat([new_df, dfOneHot], axis = 1)

## Prepare final dataset------------------------------------------------###
final_df = new_df_onehot.drop(columns=['index']).reset_index()
### Change type---------------------------------------------------------###
final_df.alley = final_df.alley.astype(np.int)
final_df.floor = final_df.floor.astype(np.int)
final_df.bhx_1km = final_df.bhx_1km.astype('float')
final_df.vin_1km = final_df.vin_1km.astype('float')
final_df.market_1km = final_df.market_1km.astype('float')
final_df.hospital_1km = final_df.hospital_1km.astype('float')
final_df.school_1km = final_df.school_1km.astype('float')
### Extract final features-----------------------------------------------###
input_variable = ['alley', 'scaled_log_floor', 
                  'scaled_squared_m2', 'scaled_log_b_room', 
                  'scaled_bhx_1km', 'scaled_vin_1km', 'scaled_market_1km', 'scaled_hospital_1km', 'scaled_school_1km',
                  'scaled_hospital_distance']
for i in dfOneHot.columns.tolist():
    input_variable.append(i)
final_df = final_df[['price'] + input_variable]

# Build website----------------------------------------------------------#
## Menu-----------------------------------------------------------------##
menu = ['Introduction', 'Prediction', 'About']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Introduction':
    st.title("Project: House Price Prediction")
    st.text('')
    st.text("Học viên thực hiện: Hà Vũ Đình Khánh")
    st.text('Giáo viên hướng dẫn: Nguyễn Quan Liêm')
    
    st.subheader('Purpose')
    st.text('''Based on the expected location and desired features, 
estimated price will be predicted and it will act as a benchmark for users
to decide whether the price is reasonable and affordable.

Moreover, it also help sellers to config features that matter to the price 
for a better price''')

    st.subheader('Target audience')
    st.text('''- Buyers who need a house for sheltering
- Sellers who want to dispose of the real estate to rellocation
- Brokers who need competitive price than other brokers to quickly sell the real estate''')

elif choice == 'Prediction':
    
    # Load model
    filename = 'poisson_glm.sav'
    poisson_modl = joblib.load(filename)
    
    st.title('Configuration')
    st.header('Desire your home :)')

    col1, col2, col3 = st.columns(3)

    #administrative_location = st.expander(label='Adminstrative Location Selection')
    # District - List
    district = col1.radio(
        "What's is your favourit district/town",
        ('district 1', 'district 2', 'district 3', 'district 4', 'district 5', 'district 6', 
        'district 7', 'district 8', 'district 9', 'district 10', 'district 11', 'district 12',
        'binh tan', 'binh thanh', 'go vap', 'phu nhuan', 'tan binh', 'tan phu', 'thu duc',
        'binh chanh (town)', 'can gio (town)', 'cu chi (town)', 'hoc mon (town)', 'nha be (town)'))
    
    # Ward
    if district == 'district 1':
        pr_district = 'quận 1'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 2':
        pr_district = 'quận 2'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 3':
        pr_district = 'quận 3'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 4':
        pr_district = 'quận 4'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 5':
        pr_district = 'quận 5'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 6':
        pr_district = 'quận 6'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 7':
        pr_district = 'quận 7'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 8':
        pr_district = 'quận 8'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 9':
        pr_district = 'quận 9'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 10':
        pr_district = 'quận 10'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 11':
        pr_district = 'quận 11'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'district 12':
        pr_district = 'quận 12'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'binh tan':
        pr_district = 'quận bình tân'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'binh thanh':
        pr_district = 'quận bình thạnh'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'go vap':
        pr_district = 'quận gò gấp'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'phu nhuan':
        pr_district = 'quận phú nhuận'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'tan binh':
        pr_district = 'quận tân bình'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'tan phu':
        pr_district = 'quận tân phú'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'thu duc':
        pr_district = 'quận thủ đức'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'binh chanh (town)':
        pr_district = 'huyện bình chánh'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'can gio (town)':
        pr_district = 'huyện cần giờ'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'cu chi (town)':
        pr_district = 'huyện củ chi'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'hoc mon (town)':
        pr_district = 'huyện hóc môn'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))
    elif district == 'nha be (town)':
        pr_district = 'huyện nhà bè'
        pr_ward = col2.radio('Please specific your desired ward:',
                        funtions.create_ward_tuple(di_wa_df, pr_district, 'ward'))

    # Alley - Radio
    alley = col2.radio(
        "What is your preference between alley and main_road?",
        ('alley', 'main road'))
    if alley == "alley":
        pr_alley = int(1)
    else:
        pr_alley = int(0)

    # Squared_m2 - Slider
    s_m2 = col3.slider('Your house squared meter', 0, 150, 60)
    pr_squared_m2 = float(s_m2)

    # Floor
    flr = col3.slider('How many floors do you want?', 0, 15, 3)
    pr_floor = float(flr)

    # Bedroom
    br = col3.slider('How many bedrooms do you want?', 0, 15, 6)
    pr_bedroom = float(br)

    st.text('')
    st.subheader('How about utilities neaby?????')
    # BachHoaXanh
    bhx = st.slider('BHX nearby?', 0, 5, 2)
    pr_bhx = int(bhx)

    # Vinmart+
    vin = st.slider('Vinmart+ nearby?', 0, 5, 1)
    pr_vin = int(vin)

    # Market
    mark = st.slider('Tradditional market nearby?', 0, 3, 1)
    pr_market = int(mark)

    # Hospital
    hosp = st.slider('Hospital nearby?', 0, 3, 1)
    pr_hospital = int(hosp)

    # Education
    scho = st.slider('Education instituion nearby?', 0, 3, 1)
    pr_school = int(scho)

    # Hospital distance
    hosp_distance = st.slider('Distance to the nearest hospital', 0.0, 5.0, 0.8, 0.01)
    pr_hospital_distance = float(hosp_distance)

    # Make predictions
    st.text('You could always change your needs!!!')
    st.subheader('Your predictive results')
    funtions.make_prediction(poisson_modl, min_max, dfOneHot, pr_district, pr_ward, pr_alley, pr_squared_m2,
                    pr_floor, pr_bedroom, pr_bhx, pr_vin, pr_market, pr_hospital, pr_school,
                    pr_hospital_distance)

elif choice == 'About':
    st.subheader('Source Data')
    st.text('[propzy.vn](https://www.propzy.vn)')
    st.text('[batdongsan.vn](https://www.batdonsan.com)')
    st.text('[propzy.vn](https://www.mogi.vn)')
