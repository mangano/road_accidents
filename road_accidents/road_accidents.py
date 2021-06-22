# General imports
from IPython.display import display, HTML, Markdown
import math 
from datetime import datetime
from datetime import date
import pickle
import random 
import collections

# General data analysis imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Models
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
#from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# My own code
import road_accidents.bplot as bplot
import road_accidents.bplot as bplot

###################################################################
#  _____                                            _             
# |  __ \                                          (_)            
# | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
# |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
# | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
# |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                | |                                         __/ |
#                |_|                                        |___/ 
####################################################################

def pre_processing(df_raw, keep_police_force=False):
    df = df_raw.copy()
    
    # define target variable
    df['y'] = df['Did_Police_Officer_Attend_Scene_of_Accident'].apply(lambda x: 0 if x!=1 else x)
    df.drop('Did_Police_Officer_Attend_Scene_of_Accident', axis=1, inplace=True)

    # accident_index is just an unique identifier
    df.drop('Accident_Index', axis=1, inplace=True)
    
    # for the moment, drop variables that are more complicated to use (geolocation)
    to_drop = ['Location_Easting_OSGR', 'Location_Northing_OSGR',
               'Longitude', 'Latitude']
    df.drop(to_drop, axis=1, inplace=True)
    
    # for the moment, drop variables that are more complicated to use (high cardinality)
    to_drop = ['Police_Force', 'Local_Authority_(District)',
               'Local_Authority_(Highway)', '1st_Road_Number', '2nd_Road_Number',
               'LSOA_of_Accident_Location'
              ]
    if(keep_police_force):
        to_drop.remove('Police_Force')
    df.drop(to_drop, axis=1, inplace=True)


    # extract month from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)

    # extract hour from Time
    df['hour'] = pd.to_datetime(df['Time']).dt.hour
    df.drop('Time', axis=1, inplace=True)
    df['hour'] = df['hour'].fillna(-1)
    
    return df
    
#####################################################################################################
#  ______            _                 _                                          _           _     
# |  ____|          | |               | |                       /\               | |         (_)    
# | |__  __  ___ __ | | ___  _ __ __ _| |_ ___  _ __ _   _     /  \   _ __   __ _| |_   _ ___ _ ___ 
# |  __| \ \/ / '_ \| |/ _ \| '__/ _` | __/ _ \| '__| | | |   / /\ \ | '_ \ / _` | | | | / __| / __|
# | |____ >  <| |_) | | (_) | | | (_| | || (_) | |  | |_| |  / ____ \| | | | (_| | | |_| \__ \ \__ \
# |______/_/\_\ .__/|_|\___/|_|  \__,_|\__\___/|_|   \__, | /_/    \_\_| |_|\__,_|_|\__, |___/_|___/
#             | |                                     __/ |                          __/ |          
#             |_|                                    |___/                          |___/           
#####################################################################################################

def plot_one_var_vs_y(var, y_true, y_false, bins):
    #print (f'### {var}')
    inputs = [y_true[var], y_false[var]]
    bplot.py_histo(inputs, labels=['y_true', 'y_false'], bins=bins, title=var)
    
def single_variables_plots(df_input):
    y_true  = df_input[df_input['y']==1]
    y_false = df_input[df_input['y']==0]
    
    bins_1_10 = np.linspace(0.5,10.5,11)
    bins_0_10 = np.linspace(-0.5,10.5,12)
    bins_m1_10 = np.linspace(-1.5,10.5,13)
    
    plot_one_var_vs_y('Accident_Severity', y_true, y_false, np.linspace(0.5,3.5,4))
    plot_one_var_vs_y('Number_of_Vehicles', y_true, y_false, np.linspace(0.5,5.5,6))
    plot_one_var_vs_y('Number_of_Casualties', y_true, y_false, bins_1_10)
    plot_one_var_vs_y('Day_of_Week', y_true, y_false, bins_1_10)
    plot_one_var_vs_y('1st_Road_Class', y_true, y_false, bins_1_10)
    plot_one_var_vs_y('2nd_Road_Class', y_true, y_false, bins_1_10)
    plot_one_var_vs_y('Road_Type', y_true, y_false, bins_1_10) 
    plot_one_var_vs_y('Speed_limit', y_true, y_false, np.linspace(-5,85,10))    
    plot_one_var_vs_y('Junction_Detail', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Junction_Control', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Pedestrian_Crossing-Human_Control', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Pedestrian_Crossing-Physical_Facilities', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Light_Conditions', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Weather_Conditions', y_true, y_false, bins_1_10) 
    plot_one_var_vs_y('Road_Surface_Conditions', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Special_Conditions_at_Site', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Carriageway_Hazards', y_true, y_false, bins_m1_10) 
    plot_one_var_vs_y('Urban_or_Rural_Area', y_true, y_false, np.linspace(0.5,3.5,4)) 
    plot_one_var_vs_y('month', y_true, y_false, np.linspace(0.5,12.5,13) )
    plot_one_var_vs_y('hour', y_true, y_false, np.linspace(-1.5,23.5,26))

    
    
#################################################
#  __  __           _      _ _             
# |  \/  |         | |    | (_)            
# | \  / | ___   __| | ___| |_ _ __   __ _ 
# | |\/| |/ _ \ / _` |/ _ \ | | '_ \ / _` |
# | |  | | (_) | (_| |  __/ | | | | | (_| |
# |_|  |_|\___/ \__,_|\___|_|_|_| |_|\__, |
#                                     __/ |
#                                    |___/ 
#################################################


def do_all(df_input):
    x_input = df_input.drop('y', axis=1).copy()
    y_input = df_input['y'].copy()

    x_train, x_test, y_train, y_test = \
        train_test_split(x_input, y_input, test_size=0.20, random_state=1)
    
    # Specify what columns are categorical
    cols = list(x_train.columns)
    categorical_indexs = []
    for x in ['1st_Road_Class', '2nd_Road_Class', 'Road_Type', 
              'Junction_Detail', 'Junction_Control',
              'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities',
              'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 
              'Special_Conditions_at_Site', 'Carriageway_Hazards'
             ]:
        categorical_indexs.append(cols.index(x))
        
    # models fitting
    models = collections.OrderedDict()
    predictions = collections.OrderedDict()

    logisticRegr = LogisticRegression(solver='liblinear')
    logisticRegr.fit(x_train, y_train)
    models['Logistic Regression'] = logisticRegr

    gbc = GradientBoostingClassifier()
    gbc.fit(x_train, y_train)
    models['Gradient Boosting'] = gbc

    lgbm = LGBMClassifier()
    lgbm.fit(x_train, y_train)
    models['Lgbm'] = lgbm

    #lgbm2 = LGBMClassifier()
    #lgbm2.fit(x_train, y_train, categorical_feature=categorical_indexs)
    #models['Lgbm2'] = lgbm2

    
    #lgbm2 = LGBMClassifier()
    #lgb_data = lgbm2.Dataset(x_train, y_train, 
    #                         categorical_feature=categorical_indexs)
    #lgbm2.train({}, lgb_data)
    #models['Lgbm2'] = lgbm2

    for key in models:
        predictions[key]=models[key].predict_proba(x_test)[:,1]
    
    
    # models evaluation
    display(Markdown('#### ROC curve:'))
    evaluate(predictions, y_test)

    # feature importance
    model = models['Lgbm']
    cols = x_test.columns
    display(Markdown('#### Features importance:'))
    get_model_features_importance(model, input_labels=cols)
    
    
    # feature importance with input shuffling
    #model = models['Lgbm2']
    #influences_df =    input_shuffling(x_data=x_test, model=model, 
    #                                   vars_to_shuffle=x_test.columns, 
    #                                   n_shuffles=50, 
    #                                   verbose=True)
    
    #display(Markdown('#### Features importance:'))
    #plot_features_importance(influences_df)
    
    return models, x_train, y_train, x_test, y_test
    
def evaluate(predictions, y_test, colors=['b', 'g', 'm', 'c'], display_wp_for='Lgbm'):

    fig, axis = plt.subplots(1,figsize=(8,8))
    axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random selection', alpha=.8)
    

    color_iterator=0
    for key in predictions:
        fpr, tpr, thr = roc_curve(y_test, predictions[key])        
        model_auc = auc(fpr, tpr)
        label = key+' model, AUC='+str(round(model_auc,3))
        axis.plot(fpr, tpr, alpha=0.8, label=label, lw=2, color=colors[color_iterator])
        color_iterator += 1
        
    axis.set_xlabel('FP Rate', fontsize=15)
    axis.set_ylabel('TP Rate', fontsize=15)
    axis.legend(loc="lower right")
    plt.show()

    ## This for the model that we specify in conf
    fpr, tpr, thr = roc_curve(y_test, predictions[display_wp_for])
    display_working_points(fpr, tpr, thr, display_wp_for)



def input_shuffling(x_data, model, vars_to_shuffle=['days_since_last_scan'],                    
                    do_sort=True,
                    n_shuffles=50, verbose=False):
    influences = collections.OrderedDict()
    for x in vars_to_shuffle:
        if(verbose):
            print ('Shuffling '+x)
        x_data_loc = x_data.copy()
    
        preds_from_shuffles = collections.OrderedDict()  
        for n in range(n_shuffles):
            x_data_loc[x] = x_data_loc[x].sample(frac=1,random_state=n).values
            preds_from_shuffles[n]=model.predict_proba(x_data_loc)[:,1]

        # tmp dataframe to faciliatete aggregation over all shuffles
        tmp_df = pd.DataFrame(preds_from_shuffles)
        influences[x] = tmp_df.std(axis=1).mean()

    # List and sort inputs importance in a compact dataframe    
    influences_df = pd.DataFrame(data=list(influences.values()), index=list(influences.keys()))
    influences_df.columns = ['importance']
    if(do_sort):
        influences_df=influences_df.sort_values('importance', ascending=False)
    return influences_df

def plot_features_importance(rankings, col_name='importance'):
    fig, axis = plt.subplots(1,figsize=(8, 6))
    axis.barh(rankings.index, rankings[col_name])

    # This is necessary to force matplotlib to order bars in the way we want and not alphabetically
    plt.yticks([x for x in range(rankings.shape[0])], rankings.index)
    axis.set_xlabel('Feature importance (arbitrary units)', fontsize=15)
    fig.autofmt_xdate()
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    axis.invert_yaxis()  # labels read top-to-bottom

def get_model_features_importance(model, input_labels):
    feature_ranking = pd.DataFrame(model.feature_importances_, input_labels)
    feature_ranking.columns = ['importance']
    df_ranking = feature_ranking.sort_values(by='importance', ascending=False)
    plot_features_importance(df_ranking, col_name='importance')
    

    
    
def display_working_points(fpr, tpr, thr, algo):
    df = pd.DataFrame({'fpr':fpr, 'tpr':tpr,'thr':thr})
    fpr_list = []
    tpr_list = []
    thr_list = []
    for fpr_thr in [0.10, 0.20, 0.30, 0.4]:
        row = df[df['fpr']>fpr_thr].sort_values(by='fpr', ascending=True).iloc[0]
        fpr_list.append(row['fpr']*100)
        tpr_list.append(row['tpr']*100)
        thr_list.append(row['thr'])

    df_2 = pd.DataFrame({'prob threshold':thr_list, 'false positive rate [%]':fpr_list,
                         'true positive rate [%]':tpr_list})
    df_2 = df_2[['prob threshold', 'false positive rate [%]', 'true positive rate [%]']]
    df_2 = df_2.round({'prob threshold':3, 'false positive rate [%]':1, 
                       'true positive rate [%]':1})
    display(Markdown('#### List of working points for model '+algo+':'))
    display(HTML(df_2.to_html(index=False)))
    
##########################################################################
#  _____                                  ____        _               _   
# |  __ \                                / __ \      | |             | |  
# | |__) | __ ___ _ __   __ _ _ __ ___  | |  | |_   _| |_ _ __  _   _| |_ 
# |  ___/ '__/ _ \ '_ \ / _` | '__/ _ \ | |  | | | | | __| '_ \| | | | __|
# | |   | | |  __/ |_) | (_| | | |  __/ | |__| | |_| | |_| |_) | |_| | |_ 
# |_|   |_|  \___| .__/ \__,_|_|  \___|  \____/ \__,_|\__| .__/ \__,_|\__|
#                | |                                     | |              
#                |_|                                     |_|              
###########################################################################


        
            
### Utilities ###


def write_pickle(filename, obj):
    outfile = open(filename,'wb')
    pickle.dump(obj, outfile)
    outfile.close()

def read_pickle(filename):
    infile = open(filename,'rb')
    object = pickle.load(infile)
    infile.close()
    return object




