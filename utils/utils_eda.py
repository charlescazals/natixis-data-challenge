"""Module contains functions to perform the EDA"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import seaborn as sns; sns.set
from scipy import stats
from scipy.stats import norm, skew


def namestr(obj, namespace):
    #function that returns the name of a variable as a string
    
    return [name for name in namespace if namespace[name] is obj]


def study_dataset(df):
    # function that studies the volatilities of the time Series. It plots an histogram as well as tries to fit a 
    # normal distribution in order to check if the standard deviations are distributed randomly 
    
    print('Studying ' + namestr(df, globals())[0])
    print('-'*40)
    centered_df = df-df.mean()
    df_info = centered_df.describe().T
    
    sns.distplot(df_info['std'] , fit=norm);
    (mu, sigma) = norm.fit(df_info['std'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('standard deviations for : ' + namestr(df, globals())[0])
    plt.show()
    
def switch_demo(argument):
    #switcher
    
    switcher = {
        'df_stock': [0,1000],
        'df_bond': [0,40],
        'df_xchang': [0,100],
        'df_yieldc': [0,6],
        'df_commod': [0,0.3],
        'df_cdsb': [0,7]
    }
    return switcher.get(argument)[0], switcher.get(argument)[1]

def study_dataset_crop(df):
    #Same function as above but cropping exgtreme values    

    print('Studying ' + namestr(df, globals())[0])
    print('-'*40)
    centered_df = df-df.mean()
    df_inf = centered_df.describe().T
    
    vmin, vmax = switch_demo(namestr(df, globals())[0])
    
    df_info = df_inf[df_inf['std'].between(vmin, vmax)]
    
    sns.distplot(df_info['std'] , fit=norm);
    (mu, sigma) = norm.fit(df_info['std'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('standard deviations for : ' + namestr(df, globals())[0])
    plt.show()
    
def missing_values(df):
    # function that studies the missing values of the time Series. It plots an histogram as well as tries to fit a 
    # normal distribution in order to check if they are distributed randomly
    
    missing_df = {}
    for col in df.columns:
        value = df[col].isna().sum() - df.reset_index()[col].first_valid_index()
        missing_df[col] = value
    
    print('Studying: ' + namestr(df, globals())[0])
    print('-'*40)
    print('Total number of series: '+ str(len(missing_df.keys())))
    print('Number of missing values on average: '+ str(sum(missing_df.values())/len(missing_df.values())))
    print('Total nb of missing values: ' + str(sum(missing_df.values())))
    print(' ')
    sns.distplot(list(missing_df.values()), fit=norm);
    (mu, sigma) = norm.fit(list(missing_df.values()))
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    
    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('Missing values distribution: ' + namestr(df, globals())[0])
    plt.show()
    
def switch_demoo(argument):
    #switcher
    
    switcher = {
        'df_stock': [0,1000],
        'df_bond': [0,2000],
        'df_xchang': [0,1500],
        'df_yieldc': [0,500],
        'df_commod': [0,2000],
        'df_cdsb': [0,500]
    }
    return switcher.get(argument)[0], switcher.get(argument)[1]

def missing_values_crop(dff):
    # Same function as above but without extreme values
    
    vmin, vmax = switch_demoo(namestr(dff, globals())[0])
    df= dff[dff['std'].between(vmin, vmax)]
    missing_df = df.isna().sum()
    info = missing_df.to_frame().describe().T
    dff = missing_df.to_frame()
    
    print('Studying: ' + namestr(df, globals())[0])
    print('-'*40)
    print('Total number of stocks: '+ str(dff.shape[0]))
    print('Number of missing values on average: '+ str(info['mean'].values[0]))
    print('Standard deviation: '+ str(info['std'].values[0]))
    print('Min: '+ str(info['min'].values[0]))
    print('Max: '+ str(info['max'].values[0]))

    print(' ')
    sns.distplot(dff[0] , fit=norm);
    (mu, sigma) = norm.fit(dff[0])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('Missing values distribution: ' + namestr(df, globals())[0])
    plt.show()