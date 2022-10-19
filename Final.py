#!/usr/bin/env python
# coding: utf-8

# In[3]:
from io import StringIO

import requests
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import json
import re
from PIL import Image



abstract = "The high proportion of energy consumed in buildings has engendered the manifestation of many environmental " \
        "problems which deploy adverse impacts on the existence of mankind. The prediction of building energy use is " \
        "essentially proclaimed to be a method for energy conservation and improved decision-making towards decreasing " \
        "energy usage. Also, the construction of energy efficient buildings will aid the reduction of total energy " \
        "consumed in newly constructed buildings. Machine Learning (ML) method is recognised as the best suited " \
        "approach for producing desired outcomes in prediction task."

motivation = "##### We want to raise awareness of the energy consumption trend of buildings!\n\n Buildings are a " \
             "central part of our daily lives, and we spend a large part of our days in them - at home, at work, " \
             "or during our spare time. In its different forms - homes, work places, schools, hospitals, libraries or " \
             "other public buildings - the built environment is, however, the single largest energy consumer. " \
             "With better estimates of these energy-saving investments, more investments in " \
             "this area will be initialized to enable progress in building efficiencies. This results in " \
             "more proficient calculation of operation costs around buildings as well as contributes to the global fight " \
             "against energy waste."

problem_statement = "Significant investments are being made to improve building efficiencies to reduce costs and " \
                    "emissions. The question is:\n\n" \
                    "**Are the improvements working?**\n\n" \
                    " We are going to present an accurate " \
                    "model of metered building energy usage in the following areas: chilled water, electric, " \
                    "hot water, and steam meters. The model for this task will be a linear regression model."

about_the_dataset = "\* Note: This dataframe is produced after merging three other dataframe, Read further to know " \
                    "more.\n\n ### About the dataset\n **The dataset includes three years of hourly meter readings " \
                    "from over one thousand buildings at several different sites around the world.**\n\n ### Files\n " \
                    "**train.csv**\n - `building_id` - Foreign key for the building metadata.\n - `meter` - The meter " \
                    "id code. Read as `{0: electricity, 1: chilledwater, 2: steam, 3: hotwater}`. Not every building " \
                    "has all meter types.\n - `timestamp` - When the measurement was taken\n - `meter_reading` - The " \
                    "target variable. Energy consumption in kWh (or equivalent). Note that this is real data with " \
                    "measurement error, which we expect will impose a baseline level of modelling error**\n - `site_id` " \
                    "- Foreign key for the weather files.\n - `building_id` - Foreign key for `training.csv`\n " \
                    "- `primary_use` - Indicator of the primary category of activities for the building\n " \
                    "- `square_feet` - Gross floor area of the building\n - `year_built` - Year building was opened\n " \
                    "- `floor_count` - Number of floors of the building\n **weather_[train/test].csv**\n\n " \
                    "Weather data from a meteorological station as close as possible to the site.\n - `site_id`\n " \
                    "- `air_temperature` - Degrees Celsius\n - `cloud_coverage` - Portion of the sky covered in " \
                    "clouds, in [oktas](https://en.wikipedia.org/wiki/Okta)\n - `dew_temperature` - Degrees Celsius\n " \
                    "- `precip_depth_1_hr` - Millimetres\n - `sea_level_pressure` - Millibar/hectopascals\n " \
                    "- `wind_direction` - Compass direction (0-360)\n - `wind_speed` - Meters per second\n "

feature_engineering = "### Feature engineering techniques in time series problem\n\n Feature engineering is the core " \
                      "part of machine learning which cannot be overlooked. With Good features, Even simplest of model " \
                      "can go magic. Feature engineering is the process of creating new features by extracting " \
                      "information from existing features, combining multiple existing features like (summing, " \
                      "multiplication, min, max, mean, standard deviation etc).\n\n While engineering features for time " \
                      "series data, we have to be careful, because features are dependent on time and hence, engineered " \
                      "feature will also depend on time.\n\n Here are the what we will use to generate new features " \
                      "based on existing one:\n\n #### Time based features\n Time based features are features extracted " \
                      "from time (usually timestamp) like day, day of year, weekend, month, hour etc. These features " \
                      "are important to generate other features based on time (eg sale on weekend, is day a holiday? etc)\n " \
                      "Here are list of available features that we could parse from timestamp using pandas.\n" \
                      "![feature engineering time series](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/" \
                      "time-features.png)\n\n We parse the DateTime object to get the DateTime feature like day, " \
                      "weekend, month, etc. Moreover, we added is_holiday a feature (American holidays in 2016, 17, 18) " \
                      "which helps us to reduce overfitting a bit."

preprocessing = "### Dealing with missing values\n\n The columns year_built and floor_count has a too high percentage " \
                "of missing values. The mean/median could corrupt the true distribution, therefore these values will " \
                "be dropped. For features with a percentage of missing values less than 50% we fill it using the " \
                "median else remove the missing value rows."


# load the sample train data
df_train_sample = pd.read_csv("./train_sample.csv")

# load column object
with open("column_list.pkl", 'rb') as output:
    column_list = pickle.load(output)

# load the texts from the json file
with open("data.json", 'r', encoding="utf8") as fp:
    text_dict = json.load(fp)

# load unique values of "primary_use" column
with open("building_primary_use_unique.pkl", 'rb') as output:
        building_primary_use_unique = pickle.load(output)
    
# load meter type dictionary object for each building
with open("building_meter_type_dict.pkl", 'rb') as output:
        building_meter_type_dict = pickle.load(output)

def final(X, labels=False):
    """
    A single function to make prediction or to compute rmsle error
    """
    # if X is a single observation, add one more dimension to match input shape
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis= 0)
    
    # convert to dataframe
    df = pd.DataFrame(X, columns= column_list)
    
    # convert columns to respective object types
    dtype_dict = {'building_id': np.int16, 
                'meter': np.int8,
                'site_id': np.int8,
                'primary_use': object,
                'square_feet': np.int32,
                'air_temperature': np.float16,
                'cloud_coverage': np.float16,
                'dew_temperature': np.float16,
                'precip_depth_1_hr': np.float16,
                'sea_level_pressure': np.float16,
                'wind_direction': np.float16,
                'wind_speed': np.float16
               }
    df = df.astype(dtype_dict)
                    
    # check if input features has right range of values
    
    # if not, return False
    if not check_input_range(df):
        return False
    
    # check if provided building has valid meter type
    if not check_buidling_meter(df):
        return False
    
    # number of provided datapoints
    n = df.shape[0]
    
    # label encoder non-numeric variable
    with open("building_primary_use_unique.pkl", 'rb') as output:
        building_primary_use_unique = pickle.load(output)
        le = LabelEncoder()
        le.fit(building_primary_use_unique)
        df.primary_use = le.transform(df.primary_use)
        
    
    # apply log1p to the area (making it more normal and dealing with extreme values)
    df.square_feet = np.log1p(df.square_feet)
    
    # change the dataformat to ease the operations
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    
    # Add date time features by parsing timestamp
    # add dayofyear column
    df["dayofyear"] = df.timestamp.dt.dayofyear
    # add day column
    df["day"] = df.timestamp.dt.day
    # add week column
    df["weekday"] = df.timestamp.dt.weekday
    # add hour column
    df["hour"] = df.timestamp.dt.hour
    # add month column
    df["month"] = df.timestamp.dt.month
    # add weekend column
    df["weekend"] = df.timestamp.dt.weekday.apply(lambda x: 0 if x <5 else 1)

    # ***************************************************************************/
    # \*  Title: [3rd Place] Solution
    # \*  Author: eagle4
    # \*  Date: 2011
    # \*  Code version: N/A
    # \*  Availability: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984
    # '''
    ##############################################################

    # "It is supposed to calculate the solar horizontal radiation coming into the building"

    latitude_dict = {0 :28.5383,
                    1 :50.9097,
                    2 :33.4255,
                    3 :38.9072,
                    4 :37.8715,
                    5 :50.9097,
                    6 :40.7128,
                    7 :45.4215,
                    8 :28.5383,
                    9 :30.2672,
                    10 :40.10677,
                    11 :45.4215,
                    12 :53.3498,
                    13 :44.9375,
                    14 :38.0293,
                    15: 40.7128}

    df['latitude'] = df['site_id'].map(latitude_dict)
    df['solarHour'] = (df['hour']-12)*15 # to be removed
    df['solarDec'] = -23.45*np.cos(np.deg2rad(360*(df['day']+10)/365)) # to be removed
    df['horizsolar'] = np.cos(np.deg2rad(df['solarHour']))*np.cos(np.deg2rad(df['solarDec']))*np.cos(np.deg2rad(df['latitude'])) + np.sin(np.deg2rad(df['solarDec']))*np.sin(np.deg2rad(df['latitude']))
    df['horizsolar'] = df['horizsolar'].apply(lambda x: 0 if x <0 else x)
    
    ##############################################################
    
    # Holiday feature
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    df["is_holiday"] = df.timestamp.dt.date.astype("str").isin(holidays).astype(int)
    
    # Drop redundent columns
    
    # Drop the columns which contains lots of missing values and have less or no effect on predicting the target
    drop_features = ['floor_count', 'year_built']
    df.drop(drop_features, axis=1, inplace=True)
    
    timestamp_list = df.pop("timestamp")
    
    # load the models
    with open("final_model_list.pkl", 'rb') as output:
        model_1, model_2 = pickle.load(output)

    # make prediction
    prediction_list =  (np.expm1(model_1.predict(df, num_iteration=model_1.best_iteration))* df.square_feet)/2
    prediction_list +=  (np.expm1(model_2.predict(df, num_iteration=model_2.best_iteration)) * df.square_feet)/2
    

    # If no labels are provided, return the predictions
    if labels is False:
        # rename prediction column
        df_tmp = pd.DataFrame(prediction_list.values, columns= ["Meter Reading"])
        # add timestamp columns
        df_tmp["Timestamp"] = timestamp_list
        return df_tmp

    # If labels are provided, compute evaluation metric
    RMSLE = np.sqrt(1/n * np.sum(np.square(np.log1p(prediction_list) - np.log1p(np.squeeze(labels)))))

    # return rmsle
    return RMSLE


# In[ ]:


# Overview of competition
st.title("How much will a building consume?")
st.write(abstract)
#st.write(text_dict["ashrae_intro_text"], unsafe_allow_html= True)

# In[ ]:

st.write("## Motivation")
st.markdown(motivation)


st.write("Problem Description (A Regression Problem)")
st.markdown(problem_statement)



# explore more
st.write("## Check The Box To Explore More")


# In[ ]:


# Roadmap checkbox
if st.checkbox('Pipeline (How We Are Tackling The Problem)'):
    #st.markdown(text_dict["roadmap_text"])
    st.write("We will analyse/perform the following operations in this research project:")
    st.markdown("* Load the data"
                "\n * Feature engineer the dataset"
                "\n * Perform exploratory data analysis"
                "\n * Preprocess the data"
                "\n     * Missing values"
                "\n     * Time alignment"
                "\n     * Merge datasets"
                "\n * Make predictions")


# In[ ]:


# Overview Dataset
if st.checkbox('About The Dataset'):
    st.subheader('Sample Train Data (10 Rows)')
    st.write(df_train_sample.head(10)
    , about_the_dataset)
    #st.write(
    ## first 10 samples
    #df_train_sample.head(10)
    #, text_dict["dataset_text"],"\n\n**_For EDA, Look At [This](https://drive.google.com/file/d/1rXnK9fF8E0QDBodtlGdPKENXd1Z7jjJJ/view?usp=sharing) PDF File_**."
    #    , unsafe_allow_html= True)


# In[ ]:


#Feature engineering
if st.checkbox("Feature Engineering"):
    st.write(feature_engineering)


#EDA
if st.checkbox("Exploratory Data Analysis"):
    st.write("## Train.csv")
    st.write("We start with some basic information about the train dataset and get a first glimpse of our data. ")
    st.image(Image.open('images/basic_trainDf.PNG'), caption='Train Dataset Basic Information', width=400)
    st.write("#### Observation")
    st.markdown("* We have details about building id (a number used to identify buildings), meter type {0: electricity, "
                "1: chilledwater, 2: steam, 3: hotwater}, timestamp (time at with reading was recorded) and the meter "
                "reading (meter reading is our target variable) \n\n* We are given the timestamp, here we are dealing "
                "with time-series data \n\n* This train dataset contains meter reading entries(4 meter types) of year "
                "2016(366 days) for 1449 unique buildings per hour (total row should be 366 * 24 * 4 * 1449) = "
                "_50912064_ \n\n* We have _20,216,100_ meter reading entries. The reason why we have 20 million records "
                "and not 50 million is that not all the buildings have all types of meters. \n\n* Minimum meter reading "
                "is _0_ and maximum meter reading is _21904700.0_ (both are probably outliers) \n\n* There are a total "
                "of 16 different sites \n\n* So, we also have meter reading value \"0\", which is quite unusual. " \
                "The question is when such a meter reading 0 can occur? Turns out, there could be plenty of reasons." \
                "\n     * **Power outage**: This could be one of many reasons when we could get a meter reading of 0. "
                "Though I also believe it could be marked as  \"nan\" (missing) because there is no reading to read."
                "\n     * **Seasonal reasons**: We have 4 meter types (0:electricity, 1: chilledwater, 2: steam, 3: hotwater),"
                " If we have 0 meter reading for any among 3 meters (excluding electricity meter, because overall "
                "electricity will be used) it might because chilled water won't be used in winter season or hotwater "
                "or steam devices are not used at all in the summer season."
                "\n     * **Closed building, Under construction or At maintenance**: This could be another factor when we have a meter reading of 0. "
                "\n     * **Error in measuring instrument(error in the meter itself)**: This could be another reason, "
                "Here there is a glitch or fault in the instrument itself.")
    st.write("")
    st.write("")
    st.write("")
    st.write("We will continue by looking at the building which has a meter reading of 21904700.0 (an outlier). "
             "We identified the building with building_id = 1099, which has 2 meter types: 0 and 2. ")
    st.image(image = Image.open('images/building1099_outlier.PNG'), caption='Building where meter reading is 21904700.0')
    st.write("#### Observation")
    st.write("* For the electricity meter (meter 0), we see a normal pattern there is no suspicion.\n"
             "*  For steam meter (meter 2), Meter reading after around march is in millions, which is quite bizarre. "
             "Here we will treat it as an outlier and perform and remove it since it could heavily skew our results.")
    st.write("Additionally, we calculate the percentile to see how the meter_reading values compare in the data set.")
    st.image(image = Image.open('images/percentile_meter_reading.PNG'), caption='Meter_reading percentiles')
    st.write('**There seems to be a huge outlier in 99th percentile which we know is from building 1099 (meter type 2) '
             'having meter readings in millions.**')
    st.write("How the overall pattern of the meter_reading across all buildings looks like can be easily displayed via "
             "a heatmap. To this end, we plot heatmaps for all the meter types.")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image=Image.open('images/electricity_heatmap.png'), caption='Electricity meter_reading heatmap')
    with col2:
        st.image(image=Image.open('images/chilledwater_heatmap.png'), caption='Chilled water meter_reading heatmap')

    col3, col4 = st.columns(2)
    with col3:
        st.image(image=Image.open('images/steam_heatmap.png'), caption='Steam meter_reading heatmap')
    with col4:
        st.image(image=Image.open('images/hotwater_heatmap.png'), caption='Hot water meter_reading heatmap')

    st.write("#### Observation")
    st.write("* The above heatmap for all the meter types show the following pattern:"
                "\n* Yellow color shows the high number of zero meter reading counts "
                "\n* Vertical yellow line shows consecutive buildings having zero meter reading "
                "(*assuming building are close to each other (neighboring)) "
                "\n* Horizontal yellow line shows the same building has zero meter reading for consecutive days"
                " \n* Buildings having non-Yellow color contains no zero meter reading")
    st.write("For meter 0, there are consecutive buildings that have 0 meter reading from day 1 to day 14 "
             "(they also have the same site id (located at the same location))")
    st.write("* For meter 1, 2, and 3 (hotwater, steam, and chilled water meter), we have many horizontal yellow "
             "lines (zero meter reading), which shows either of the devices are not being used. This is normal as not "
             "many people use chilled water in the winter or the hotwater in the summer."
             "\n* Having zero values for meter reading (especially in the case of an electricity meter) could be "
             "problematic to the model while learning. We will try to remove buildings from the time frame if the "
             "meter reading is 0 for many consecutive days (treating them as outliers)")

    st.write("")
    st.write("")
    st.write("")
    st.write("Now, under the assumption that buildings close to each other share similar characteristics we investigate"
             "if there is any relationship between building_id and site_id. We will plot a regression plot to "
             "capture the relationship (if it exists).")
    st.image(image=Image.open('images/regressionplot_building_site.png'),
             caption='Relationship between building_id and site_id')
    st.write("#### Observation")
    st.markdown("There is a positive relation between site id and building id. What does it mean?\n\n"
             "It means, buildings whose ids are close, are actually close to each other (building id 45 is much closer "
             "to the building is 47 than building id 100 or building id 1.). Why this is important?\n\n "
             "If any site went down (due to some reason, maybe power outage or some natural disaster happens), "
             "there is a high probability that meter reading of buildings located at the same site will have a similar "
             "effect (because buildings are close to each other)."
             "\n\n It also seems that building ids are assigned in a sequential manner, one after another."
             "\n\nYou can also see most of the buildings are from site 3, and hence the majority of train data has "
             "come from site 3.")
    st.write("### Target Variable")
    st.write("The target variable (meter_reading) is heavily positive skewed with outliers. Large values skew the "
             "graph of the data so better plot via logarithmic scale.")
    st.image(image=Image.open('images/hit_meter_reading.png'),
             caption='Distribution of target')
    st.image(image=Image.open('images/log_meter_reading_type.png'),
             caption='Distribution of target by meter type')
    st.write("#### Observation")
    st.markdown("* All the meter types contain a high 0 meter reading value, which is ok in all other cases except for "
                "the electricity meter.")
    st.write("Let's now consider the meter_reading over a single day and then over a whole weekand see if we can learn "
             "something.")
    st.image(image=Image.open('images/meter_reading_day.png'),
             caption='Meter reading per day')
    st.image(image=Image.open('images/boxplot_meter_reading_week.png'),caption='Meter reading weekwise')
    st.write("#### Observation")
    st.markdown("* We can note a significantly higher meter reading during working hours. This could indicate a great "
                "influence on the prediction of the model."
                "\n* Sundays show the lowest meter readings of a whole week.")

    st.write("## Weather.csv")
    st.write("#### Observation")
    st.write("There are total of 139773 entries and 15 features (with feature engineering). It contains missing values.")
    st.image(image=Image.open('images/weather_missing_pct.png'), caption='Weather missing values')
    st.markdown("* There are 7 columns which contains missing values (air_temperature, cloud_coverage dew_temperature, "
                "precip_depth_1_hr, sea_level_pressure, wind_direction, wind_speed) "
                "\n * cloud_coverage has around 50% of the values that are "
                "missing, followed by precip_depth_1_hr (around 36%) of missing values. To address this issue, "
                "we will try to drop these two columns and see if it improves the metric. The rest of the missing values "
                "wewill first look at the distribution and see how we could impute these values appropriately.")
    st.write("### Feature Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image=Image.open('images/dist_airtemperature.png'), caption='Distribution air_temperature')
        st.image(image=Image.open('images/hist_recip_depth.png'), caption='Distribution precip depth 1 hour')
        st.image(image=Image.open('images/dist_dew_temperature.png'), caption='Distribution dew temperature')
        st.image(image=Image.open('images/dist_wind_speed.png'), caption='Distribution wind speed')
    with col2:
        st.image(image=Image.open('images/dist_cloudcover.png'), caption='Distribution cloud coverage')
        st.image(image=Image.open('images/dist_sea_level_pressure.png'), caption='Distribution sea level pressure')
        st.image(image=Image.open('images/dist_wind_direction.png'), caption='Distribution wind direction')

    st.write("#### Observation")
    st.write("* Apart from the features cloud_coverage, precipitaion_depth_1hr and wind_direction the weather train "
             "data features are distributed normal. This means that we can impute missing values safely with the "
             "mean/average."
             "\n * **For air temperature, there are a few drops along the distribution. We will further discuss this in "
             "the section on Preprocessing.**")
    st.write("Moving on to the building dataset...")

    st.write("## Building.csv")
    st.write("First we take a look at some basic information about the dataset.")
    st.image(image=Image.open('images/basic_building.PNG'), caption='Building dataset statistics')
    st.markdown("+ The data is collected from 16 different sites worldwide "
                "\n+ We have 1449 building samples. "
                "\n+ The square feet of the buildings ranges from 283 to 875000. The distribution is quite high. "
                "\n+ The buildings built year is ranges from 1900 to 2017. Both 19th centuary and 20th centuary "
                "buildings are present."
                "\n+ The floor_count of the buildings ranges from 1 to 26.")
    st.write("For the building dataset we also took a look at the missing values and found the following:")
    st.markdown("+ Build year seems to be missing for more than 50% of the building (around 75%), so does floor_count (around 50%). So we "
                "will drop these features and observe the metric "
                "\n+ Also, as seen in the EDA notebook, there is a relation between site_id and building_id."
                " We have to include these 2 features for sure.")

    st.write("### Feature Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image=Image.open('images/dist_square_feet.png'), caption='Distribution square feet')
        st.image(image=Image.open('images/dist_floorCount.png'), caption='Distribution floor count')
    with col2:
        st.image(image=Image.open('images/dist_year_built.png'), caption='Distribution year built')

    st.write("#### Observation")
    st.markdown("* The distribution is positively skewed. There are few buildings whose area is more than "
                "200000 square feet. To make it look more normal, we will apply log transformation while modeling the data."
                "\n* As mentioned, we have buildings from 2 different centuries with most of them considered older buildings."
                "\n* Moreover, the floor count shows a high 0 count")
    st.write("**At first we thought of dividing the meter reading by the square feet oder floor count trying to "
             "standardize the target variable, but the feature distributions are not that suitable. Also the feature "
             "floor count will be dropped.**")
    st.write("Nevertheless, we include square feet for our modeling since we noted a positive correlation with the"
             "target variable.")
    st.image(image=Image.open('images/corr_meter_reading_squarefeet.png'), caption='Correlation meter reading and square feet')
    st.write("We can clearly see a positive relation between median meter reading and square feet (in log scale). "
             "The area of the building could be helpful at the time of modeling the problem.")
    st.write("A short inspection of the primary use of a building shows us:")
    st.image(image=Image.open('images/hist_primary_use.png'),
             caption='Historgram primary use of building')
    st.write("#### Observation")
    st.write("Most of the building in this dataset is used for educational purpose, followed by office use, "
             "Entertainment/public assembly, Public services, Lodging/residential")
    st.write("")
    st.write("That concludes our exploratory data analysis on the 3 datasets. Now we move on to the preprocessing step.")

#Preprocessing
if st.checkbox("Performed Preprocessing"):
    st.write("### Dealing with missing values")
    st.write("In the building dataset the columns year_built and floor_count have a too high percentage of missing values. "
             "The mean/median could corrupt the true distribution, therefore these values will be dropped.")
    st.write("For all remaining features ff the percentage of missing values is less than 50 then fill it using the mean/median.")
    st.write("### Insights from air temperature values")
    st.write("When plotting a histogram of air Temperatures, we saw bins with fewer records. After discussing this topic"
             "in our research group, we came to the conclusion that we cannot be sure to know with which metric the temperature"
             "was measured (Fahrenheit or Celsius) and if this was considered when creating the datatset. Furthermore, "
             "the geographical location of the weather sites are not known and this could influence the timezone.")
    st.write("To this end, we look at what time of the day peak temperature occurs.")
    st.image(image=Image.open('images/heatmap_timestamp_misaligned.png'),
             caption='Temperature peaking times')
    st.write("Sometimes temperature peaks in the evening, which implies that the time for temperature date is not local.")
    st.write("From the above heatmap we note 3 different types values for the hour of day with highest average temperature:")
    st.markdown("* **19:00-20:00-21:00:** site ids 0, 3, 6 , 7, 8, 9, 11, 13, 14, 15"
                "\n* **14:00:** 1, 5, 12"
                "\n* **22:00-23:00:** 2, 4, 10")
    st.write("We guess that the timestamp is UTC or something close by. The difference between 19:00 and 22:00 could "
             "come from US time zones. Potentially the other site ids could be timezones as USA but different countries.")
    st.write("**Then, assuming (1,5,12) tuple has the most correct temp peaks at 14:00, we can calculate offsets for "
             "the other site ids and align them accordingly**")
    st.image(image=Image.open('images/heatmap_timestamp_aligned.png'),
             caption='Aligned temperature peaking times')

    st.write("### Encode categorical features")
    st.write("We have only one categorical column out of all the datasets and that is in building dataset. "
             "The primary_use column in the building dataset is object type. There are different techniques to "
             "convert the string type columns into numerical. Since one hot encoding technique increases the number of "
             "columns and tend to increase the size of the dataset, which led use ‘LabelEncoder’ to convert strings "
             "to numericals.")

    st.write("That is all for the preprocessing part.")


# Overview modeling
if st.checkbox('Input Format Instructions'):
    st.write('#### Sample Train Data (10 Rows)')
    st.write(
    # first 10 samples
    df_train_sample.head(10))
    
    st.write(
        text_dict["input_format_text"], 
        unsafe_allow_html= True
    )


# In[ ]:


# Prediction And Evaluation
submit_button_object = None
if st.checkbox('Prediction And Evaluation'):
    st.write("### Get Prediction (No Labels Required) Or Compute Error Metric (Require Labels)")
    st.write("#### Enter The Input Data Here:")
    st.write(f"Column Sequence: `{np.array(column_list)}`", "\n\n**\*Note**: Not all the building has all the meter type.")
    user_input = st.text_area('', '''[789, 2, 2017-1-26 01:00:00, 7, Education, 64583, 1923.0, 1.0, -10.0, nan, -13.5, nan, 1026.0, 70.0, 4.6]''')
    submit_button_object = st.button('Make Prediction/ Compute RMSLE')
    #render_graph = st.checkbox('Draw Graph (works when only features are provided)')
    render_graph = False

# In[ ]:


# convert string list to numpy array object
def string_to_list(X, y= False):
    '''Convert string list to list/numpy object'''
    # process features
    features = re.findall(r"(\[[^\[].+?\])",X)
    
    for idx, row in enumerate(features):
        # strip all spaces and convert to numpy array object
        features[idx] = np.array([element.strip() for element in row.strip('][').split(',')])
        
        # if a row doesn't have 15 features, ask user to provide it
        if len(features[idx]) != 15:
            st.error(f"Row {idx + 1} contains {len(features[idx])} features whereas there should be 15 features")
            return False
    
    features = np.array(features)
    
    # process labels and validate "labels" section
    if y:
        # parse string to list object
        labels = y.strip('][').split(',')
        
        # if we fail to convert to numpy array, invalid format provided
        try:
            labels = np.array(labels)
        except:
            # tell user to provide valid input for labels
            st.error("**Invalid label format provided**:\n\n Valid format: **[label_row_1, label_row_2, label_row_3]**")
            # return False
            return False
        
        # if non valid datatype if provided, alert the user
        try:
            labels = np.float16(labels)
        except:
            # ask the user to provide valid input format
            st.error("**Empty or Invalid Label Type Is Provided**:\n\n Valid Datatype: **Float**")
            # return False
            return False
        
        # make sure feature and labels have same 1st dimension in shape
        if labels.shape[0] != features.shape[0]:
            # alert the user to provide same number of rows
            st.error("**Number of feature rows does not match the number of label rows provided.**\n\nMake Sure To Provide Label For Each Feature Row")
            # return False
            return False
        return features, labels
    
    # return only features
    return features


# In[2]:


def check_input_range(df):
    '''Check if user inputs is in right range'''
    
    # check discrete numeric type
    # ['building_id', 'meter', 'site_id', 'primary_use']
    disc_col_dict = {0: ('building_id', [1, 1449]), 1: ('meter', [0, 3]), 3: ('site_id', [0, 15])}
    
    # for each discrete column, check the input value range
    for col_idx, disc_col_tuple in disc_col_dict.items():
        col_name = disc_col_tuple[0]
        lo = disc_col_tuple[1][0]
        hi = disc_col_tuple[1][1]
        
        if not df[col_name].between(lo, hi).all():
            st.error(f"**Value Out Of Range for column {col_idx + 1} ('{col_name}')**.\n\n Valid values should be within(inclusive): **[{lo} - {hi}]**")
            return False
    
    # check value set for primary_use column
    col_idx = 4
    col_name = 'primary_use'
    if not df[col_name].isin(building_primary_use_unique).all():
            st.error(f"**Value Out Of Range for column {4} ('{col_name}')**.\n\n Valid values should be one of :**{', '.join(building_primary_use_unique)}**")
            return False
    # all the inputs are in valid range and set, process the input further
    return True


# In[ ]:


def check_data_types(X):
    '''Check if provided input has valid data type'''
    
    dtype_dict = {0: ["building_id", "Integer",np.int16], 
                1: ['meter', "Integer", np.int8],
                3: ['site_id', "Integer", np.int8],
                5: ['square_feet', "Integer", np.int32],
                7: ['air_temperature', "Float", np.float16],
                8: ['cloud_coverage', "Float", np.float16],
                9: ['dew_temperature', "Float", np.float16],
                10: ['precip_depth_1_hr', "Float", np.float16],
                11: ['sea_level_pressure', "Float", np.float16],
                12: ['wind_direction', "Float", np.float16],
                13: ['wind_speed',"Float",  np.float16]
               }
    
    # for each column, convert to repsective data type
    for col_idx, list_ in dtype_dict.items():
        column_name, valid_type, dtype = list_[0], list_[1], list_[2]
        
        # try to convert current column into it's respective data type
        try:
            X[:, col_idx] = X[:, col_idx].astype(dtype)
        
        except Exception as e:
            
            st.error(f"**Invalid Data Type Provided For Column {col_idx} ('{column_name}')**.\n\n Valid data type: **{valid_type}**")
            return False
    
    
    # check for timestamp column
    col_idx = 2
    
    column_name, valid_type, dtype = "timestamp", "YYYY-MM-DD HH:MM:SS, eg ' 2017-12-31 01:00:00' ", "Datetime Format"
    
    # try to convert datetime column into datetime data format
    try:
        pd.to_datetime(X[:, col_idx], format="%Y-%m-%d %H:%M:%S")

    except Exception as e:
        st.error(f"**Invalid Datetime Format Provided For Column {col_idx} ('{column_name}')**.\n\n Valid Datetime Format: **{valid_type}**")
        return False
    
    return X


# In[ ]:


# check if meter type is present for a provided building
def check_buidling_meter(df):
    '''Not all the buildings has all the meter types, check if building has provided meter type'''
    
    # for each building, check the meter type
    for idx, lst in enumerate(df[["building_id", "meter"]].values):
        bid, meter = lst[0], lst[1]
        if meter not in building_meter_type_dict[bid]:
            
            # show the error
            st.error(f"**No Meter Type '{meter}' For Building {bid}**. Meter Should Be One Of: **{building_meter_type_dict[bid]}**")
            return False
    return True


# In[ ]:


# handle user submission
if submit_button_object:
    
    # if both features and labels are provided
    if len(user_input.split(";")) == 2:
         
        # divied features and labels
        X, y = user_input.split(";")
        
        # check if square brackets matches for input
        if X.count("[") != X.count("]") or y.count("[") != y.count("]"):
            # alert the user
            st.error("**Invalid Format Provided.**\n\nValid Format: **[[feature_row_1, feature_row_2, ...]];[label_row_1,  label_row_2, ...]**")
        
        else:
            # remove leading and trailing spaces
            X = X.strip()
            y = y.strip()

            # if either of the string is empty, print error message
            if not X.strip() and not y.strip():
                st.error("**Empty Input Provided**:\n\nMake sure both the inputs are non empty")
            
            # if no features are provided
            elif not X.strip():
                st.error("**Empty Input Provided For Features.**")
            
            # if no labels are provided
            elif not y.strip():
                st.error("**Empty Input Provided For Lables.**")

            # process input further
            else:
                # convert list string to numpy array
                return_value = string_to_list(X, y)

                # error has been encountered, do nothing
                if isinstance(return_value, bool):
                    # do nothing
                    pass

                # process input further
                else:
                    # Parse the input
                    X, y = return_value
                    
                    # if X is a single observation, add one more dimension to match input shape
                    if len(X.shape) == 1:
                        X = np.expand_dims(X, axis= 0)
                    
                    ##### check if input features has right datatype
                    dtype_checked = check_data_types(X)
                    
                    # if not, alert the user
                    if isinstance(dtype_checked, bool):
                        pass
                    # else process the input further
                    else:
                    
                        # compute the metric
                        RMSLE = final(X, y)

                        if isinstance(RMSLE, bool):
                            # do nothing
                            pass

                        # process the input further
                        else:
                            # update the user
                            st.write(f"# RMSLE: {RMSLE:.3f}")

    # if only features are provided
    elif len(user_input.split(";")) == 1:
        
        X = user_input
        
        # check if square brackets matches for input
        if X.count("[") != X.count("]"):
            # alert the user
            st.error("**Invalid format provided.**\n\nValid format: **[[feature_row_1, feature_row_2]];[label_row_1,  label_row_2]**")
        
        # remove leading and trailing spaces
        X = X.strip()

        # If input is empty
        if not X:
            st.error("**Empty Input Provided**:\n\nMake sure input is non empty")
        
        # process input further
        else:
            
            # convert list string to numpy array
            return_value =  string_to_list(X)

            # error has been encountered, do nothing
            if isinstance(return_value, bool):
                pass
            
            # proceed input further
            else:
                X = return_value
                
                # if X is a single observation, add one more dimension to match input shape
                if len(X.shape) == 1:
                    X = np.expand_dims(X, axis= 0)
                
                ##### check if input features has right datatype
                dtype_checked = check_data_types(X)
                    
                # if not, alert the user
                if isinstance(dtype_checked, bool):
                    pass
                
                # else process the input further
                else:
                    # else process the data further
                    predictions = final(X)

                    # error has been encountered, do nothing
                    if isinstance(predictions, bool):
                        # do nothing
                        pass
                    # process the input further
                    else:

                        st.write("# Prediction/s:")
                        st.write(predictions)

                        # if graph checkbox is checked, draw the graph
                        if render_graph:
                            # if X is a single observation, add one more dimension to the shape to match input shape
                            if len(X.shape) == 1:
                                X = np.expand_dims(X, axis= 0)

                            # render graph for each building id
                            for idx, data_list in enumerate(X[:,0:2]):

                                bid, meter = data_list

                                # load the data for building id, meter pair
                                with open(f"./building_meter_reading/meter_reading_bid_{bid}_meter_{meter}.pkl", 'rb') as output:
                                    df_meter_reading = pickle.load(output)


                                # plot all the target points of current buidling id and meter
                                fig = px.line(df_meter_reading, x="timestamp", y="meter_reading", title=f'Meter reading of building id "{bid}" and meter "{meter}" over time')

                                # st.write(pd.DataFrame(predictions.iloc[idx].values.reshape(1, 2), columns= ["Meter Reading", "Timestamp"]))
                                tmp = pd.DataFrame(predictions.iloc[idx].values.reshape(1, 2), columns= ["Meter Reading", "Timestamp"])
                                tmp["text"] = "prediction"
                                data = px.scatter(tmp,
                                                     x="Timestamp",
                                                     y="Meter Reading",
                                                 color= "Timestamp",
                                                 hover_name= "text")

                                # current prediction at given timestamp
                                fig.add_trace(data.data[0])

                                fig.update_layout(
                                autosize=False,
                                width=1000,
                                height=500,)

                                st.plotly_chart(fig)

    else:
        # invalid data format
        st.error("**Invalid Input Format Provided**.\n\nProvided format should be as below: eg **`[[789, 2, 2017-1-26 01:00:00, 7, Education, 64583, 1923.0, 1.0, -10.0, nan, -13.5, nan, 1026.0, 70.0, 4.6]];[600]`**")

