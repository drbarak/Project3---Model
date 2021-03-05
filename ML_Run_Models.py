#!/usr/bin/env python
# coding: utf-8

# In[1]:


# General tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import inspect
import pickle
import re

# For validation
from sklearn.model_selection import train_test_split
# For curve fitting
from scipy.optimize import curve_fit

from IPython.display import display
p = print
d = display

import dataframe_image as dfi

# In[2]:


# For transformations and predictions
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# In[4]:

class ML_model():
  def __init__(self, nround = 4):
    self.nround = 4
    
  def update(self, model, X_train, X_test, y_train, y_test, X, y, df_, is_log_scale):
    self.model = model
    self.X_train = X_train
    self.X_test = X_test 
    self.X = X
    if is_log_scale:
      self.y_train = np.expm1(y_train)
      self.y_test = np.expm1(y_test)
      self.y = None if y is None else np.expm1(y)
    else:
      self.y_train = y_train
      self.y_test = y_test
      self.y = y
    self.df_ = df_
    self.valid_rmse = np.round((df_.test_rmse / df_.train_rmse - 1) * 100, 2)
    self.valid_rmsle = np.round((df_.test_rmsle / df_.train_rmsle - 1) * 100, 2)
    
  def update_pred(self, y_train_pred, y_test_pred, is_log_scale):
    if is_log_scale:
      self.y_train_pred = np.round(np.expm1(y_train_pred), self.nround)
      self.y_test_pred = np.round(np.expm1(y_test_pred), self.nround)
    else:
      self.y_train_pred = np.round(y_train_pred, self.nround)
      self.y_test_pred = np.round(y_test_pred, self.nround)

ML = ML_model()

linearRegression, decisionTree, kNeighborsRegressor, randomForestRegressor = 0, 1, 2 ,3
kNearsetNeighbors=kNearsetNeighbours=kNeighborsRegressour=kNeighborsRegressor

models = [
    LinearRegression(), 
    DecisionTreeRegressor(), 
    KNeighborsRegressor(),      # The slowest one, by far
    RandomForestRegressor()]    # takes longer than the first 2

def model_name(model):
    return type(model).__name__


# ## Functions for calculating error

# In[5]:


# For scoring
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse

def rmse(a, b):
    return mse(a, b) ** 0.5

def rmsle(a,b):
    return msle(a,b) ** 0.5

# ## Function to score a model

# In[6]:


def score_model_(stage, y, y_pred, is_log_scale):

    if not is_log_scale:
        rmse_error = rmse(y, y_pred)
        # MSLE cannot be used with negative values
        rmsle_error = rmsle(y.clip(lower=0), y_pred.clip(min=0))   # pandas uses 'lower', numpy uses 'min'
    else:
        rmse_error = rmse(np.expm1(y), np.expm1(y_pred))
        rmsle_error = rmse(y, y_pred)

    rmsp_error = np.round(100*(np.expm1(rmsle_error)), 2)
    
    return pd.DataFrame({
        f"{stage}_rmse": np.round(rmse_error, ML.nround),
        f"{stage}_rmsle": np.round(rmsle_error, ML.nround),
        f"{stage}_rms%": rmsp_error
    }, index=[0])

def score_model(model, y_train, y_train_pred, y_test, y_test_pred, is_log_scale):
    # score the model
    results = score_model_('train', y_train, y_train_pred, is_log_scale)
    results = results.join(score_model_('test', y_test, y_test_pred, is_log_scale))
    
    results['model'] = model_name(model)
    return results.set_index('model')


# ## Function to fit, predict and scroe using each one of the models  
def disp_to_file(df, path):
  with open(path, "w") as f:
#    html = df.to_html()
#    f.write(html.replace('<tr>', '<tr align="right">'))
    df_styled = df.style
    dfi.export(df_styled, path)
    
def plot_model(ax, ML, is_log_scale, y_label=True, train=False):
  if train:
    ax.plot(ML.y_train, ML.y_train_pred, '.b')
    ax.plot(ML.y_train, ML.y_train, linewidth=3, color='g')
    ax.set_xlabel(ML.y_train.name)
    if y_label:
      ax.set_ylabel('y_train_pred')
    if is_log_scale:
      ax.set_title('{}\nTrain_rmsle={:.2f}'.format(model_name(ML.model), ML.df_.iloc[0].train_rmsle))
    else:
      ax.set_title('{}\nTrain_rmse={:.2f}'.format(model_name(ML.model), ML.df_.iloc[0].train_rmse))
  else:
    ax.plot(ML.y_test, ML.y_test_pred, '.b')
    ax.plot(ML.y_test, ML.y_test, linewidth=3, color='g')
    ax.set_xlabel(ML.y_test.name)
    if y_label:
      ax.set_ylabel('y_test_pred')
    if is_log_scale:
      ax.set_title('{}\nTest_rmsle={:.2f} Valid_rmse%={:.2f} Valid_rmsle%={:.2f}'.format(model_name(ML.model), 
                                                            ML.df_.iloc[0].test_rmsle, ML.df_.iloc[0]['valid_rmse%'], ML.df_.iloc[0]['valid_rmsle%']))
    else:
      ax.set_title('{}\nTest_rmse={:.2f} Valid_rmse%={:.2f} Valid_rmsle%={:.2f}'.format(model_name(ML.model), 
                                                            ML.df_.iloc[0].test_rmse, ML.df_.iloc[0]['valid_rmse%'], ML.df_.iloc[0]['valid_rmsle%']))#In[7]:

def model_fit_score(model, X_train, y_train, X_test, y_test, is_log_scale, ratio_col):
          # learn the model
        #p(model_name(model))
        y_fit = y_train
        if not ratio_col is None:
          y_fit = y_fit / X_train[ratio_col]

        model.fit(X_train, y_fit)

        # create train/test predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if not ratio_col is None:
          y_train_pred = y_train_pred * np.array(X_train[ratio_col])
          y_test_pred = y_test_pred * np.array(X_test[ratio_col])
          
        ML.update_pred(y_train_pred, y_test_pred, is_log_scale)

        # score the model
        return score_model(model, y_train, y_train_pred, y_test, y_test_pred, is_log_scale)

def fit_and_score(X_train, y_train, X_test, y_test, is_log_scale, start_model, max_model, ratio_col, plot, png_path, disp):
    df = pd.DataFrame()
    if plot:
      fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(12,8))
      fig.tight_layout(pad=5)   # space between subplots
      ax_n = 0
    for model in models[start_model:max_model]:
        if disp: p(model_name(model))
        results = model_fit_score(model, X_train, y_train, X_test, y_test, is_log_scale, ratio_col)
        r = results.test_rmsle / results.train_rmsle if is_log_scale else results.test_rmse / results.train_rmse
        results['valid_rmse%'] = np.round((results.test_rmse / results.train_rmse - 1) * 100, 2)
        results['valid_rmsle%'] = np.round((results.test_rmsle / results.train_rmsle - 1) * 100, 2)
        ML.update(model, X_train, X_test, y_train, y_test, None, None, results, is_log_scale)
        
        if plot:
          plot_model(axs[ax_n // 2,ax_n % 2], ML, is_log_scale, ax_n == 0)
          ax_n += 1
          
          # to save memory, since X and Y are identical, we save it in the first model only
        if model != models[start_model]:
          ML.Xtrain = ML.X_test = ML.y_train = ML.y_test = None
        results['ML'] = pickle.loads(pickle.dumps(ML))
        df = df.append(results)
    if not png_path is None:
      plt.savefig(png_path)
    plt.show()
    return df
  
# In[9]:

def split_(diamonds, is_log_scale, col, split_random_state):
    # remove all non numeric columns
  #s = diamonds.dtypes[diamonds.dtypes != 'int64'][diamonds.dtypes != 'float64'][diamonds.dtypes != 'bool']
  x = diamonds.dtypes.astype(str)
  s = diamonds.dtypes[~(x.str.contains('int|float|bool', regex=True))]
    # check if there columns that the data can be converted to numeric (such as 'category')
  for c in s.index:
    t = pd.to_numeric(diamonds[c])
    if not re.search('int|float|bool', str(t.dtype)) is None:
      s = s.drop(c)
  if len(s) > 0:
    print('NOTE: the following non-numeric columns where not included in the model:')
    print('\t',list(s.index))
    print()
    display(diamonds.head(2))
    for c in s.index:
      diamonds = diamonds.drop(c, axis=1)
  
  log_price = None
  if is_log_scale:
    if 'log_' + col in diamonds:
      col = 'log_' + col
    else:
      log_price = np.log1p(diamonds[col])
  X, y = diamonds.drop(columns=col), diamonds[col]
  if not log_price is None:
    y = log_price
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=split_random_state)
  return X_train, X_test, y_train, y_test, X, y

def display_args(func, last=None):
    spec = inspect.getfullargspec(func)
    defaults = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))
    defaults.update(spec.kwonlydefaults or {})
    if not last is None:
      defaults['**kwargs'] = '{}'
    p(defaults)
    p()
    msg = """
    model_num = one of (linearRegression, decisionTree, kNeighborsRegressor, randomForestRegressor), used for a single model run
    start_model, max_model = range of model_num, using for multiple models run (using run_models)
    col = target column name to predict
    disp = True, to display the results of the model run
    ratio_col = the column to divide the target column by, to have the model use the ratio instead of the actual value. 
          For example, if the target is 'tip', it might be better to look at the ratio with the meal price to try to predict the tip percentage.
          The output is in actual values and the conversion back and forth is done automatically
    is_log_scale = True, means:
          If a column by the name ['log_' + col] exists in the database it is assumed that it is the target column and it is already logarthmic. 
          Otherwise, the target column is converted to log and then dropped for the model. The output is in actual values and the conversion back and forth is done automatically
          In this case one should look at the rmsle result instead of rmse, and the valid% is test_rmsle/train_rmlse
    plot = True, means:
          To plot test_pred compared to actual test values (y_test_pred vs. y_test) for each model.
          If a single model (using run_model()) it displays also train_pred compared to train, side by side
    png_path = full path and file name to save the picture of the plot
    split_random_state = integer value to fix the split between train and test
    kwargs = all the parameters at the end of the parameter list are passed as hyper_parameters of the mode - used in a single model only (run_model)
    """
    p(msg)
    return
  
def run_models(diamonds=None, disp=True, is_log_scale=False, start_model=0, max_model=4, plot=False, png_path=None, ratio_col=None, col_ratio=None, ratio=None, disp_path=None,  
               *, col='price', split_random_state=314159):
  """
    keyword arguments: all arguments after the star (*) must be specified by name
        Thus, only the database name is a positional argument and all the rest must be specifed explicitly
  """
  if diamonds is None:
    return display_args(run_models)
  ratio_col = ratio_col if not ratio_col is None else col_ratio if not col_ratio is None else ratio

  X_train, X_test, y_train, y_test, X, y = split_(diamonds, is_log_scale, col, split_random_state)
  df_ = fit_and_score(X_train, y_train, X_test, y_test, is_log_scale, start_model, max_model, ratio_col, plot, png_path, disp)
  if (disp):
    d(df_[df_.columns[:-1]])
            
  if not disp_path is None:
    disp_to_file(df_[df_.columns[:-1]], disp_path) 
    
  return df_

# In[10]:

models_lookup = {0: LinearRegression, 1: DecisionTreeRegressor, 2: KNeighborsRegressor, 3: RandomForestRegressor}
    
def run_model(diamonds=None, disp=True, is_log_scale=False, plot=False, png_path=None, ratio_col=None, col_ratio=None, ratio=None, disp_path=None, 
              *, model_num=linearRegression, col='price', split_random_state=314159, **kwargs):
  """
    keyword arguments: all arguments after the star (*) must be specified by name
        Thus, only the database name is a positional argument and all the rest must be specifed explicitly
  """
  if diamonds is None:
    return display_args(run_model, '**kwargs')
  ratio_col = ratio_col if not ratio_col is None else col_ratio if not col_ratio is None else ratio
  
  X_train, X_test, y_train, y_test, X, y = split_(diamonds, is_log_scale, col, split_random_state)
  model = models_lookup[model_num](**kwargs)
  df_ = model_fit_score(model, X_train, y_train, X_test, y_test, is_log_scale, ratio_col)
  ML.update(model, X_train, X_test, y_train, y_test, X, y, df_, is_log_scale)
  df_['valid_rmse%'] = ML.valid_rmse
  df_['valid_rmsle%'] = ML.valid_rmsle
  if disp:
    d(df_)
           
  if not disp_path is None:
    disp_to_file(df_, disp_path)
    
  if plot:
    fig, axes = plt.subplots(1,2, figsize=(13, 4))
    for ax in axes:
      plot_model(ax, ML, is_log_scale, train=(ax==axes[0]))
    if not png_path is None:
      plt.savefig(png_path)
    plt.show()
  return ML

def split(X, y, test_size=None, random_state=None):
  return train_test_split(X, y, test_size=test_size, random_state=random_state)

def curvefit(func, inp, out):
  return curve_fit(func, inp, out)

pd.set_option('max_colwidth', 500)

def hp_loop(df=None, disp=False, model_num=randomForestRegressor, hp_name=None, hp_range=[], is_log_scale=True, col='SalePrice', ratio_col=None, plot=True, png_path=None, disp_path=None, split_random_state=314159, **kwargs):
  result = pd.DataFrame()
  model = models_lookup[model_num]()
  dic = model.get_params()
  if not hp_name in dic:
    p(f"Paramter '{hp_name}' not an hyperparamter of model {model_name(model)}")
    p("Possible parameters are:")
    p("\t", dic)
    return
  if not 'n_jobs' in kwargs:   # speedup the model calculation
    kwargs['n_jobs'] = -1
  p("Starting hp_loop: ", kwargs)
  for feature in hp_range:
    kwargs[hp_name] = feature
    ML = run_model(df, disp=False, model_num=model_num, is_log_scale=is_log_scale, col=col, ratio_col=ratio_col, plot=False, png_path=png_path, split_random_state=split_random_state,                                **kwargs)
          
    df_ = ML.df_.reset_index()
    df_[hp_name] = 0 if feature is None else int(feature) if type(feature) == bool else feature
    df_['hyper_parameters'] = f"{kwargs}"
    #p(feature, ':', df_.iloc[0]['test_rmsle'], ",", df_.iloc[0]['hyper_parameters'])
    result = result.append(df_.loc[:, ['train_rmsle', 'test_rms%', 'test_rmse','test_rmsle', 'valid_rmsle%', hp_name, 'hyper_parameters']],ignore_index=True)
    
  if plot:
      #d(result[[hp_name, 'test_rmsle', 'train_rmsle', 'valid_rmsle%']])
      ax = result.plot(hp_name, 'test_rmsle', marker='o', kind='scatter', figsize=(8, 5))
      result.plot(hp_name, 'train_rmsle', marker='o', c='r', kind='scatter', figsize=(8, 5), ax=ax)
      plt.legend(['test', 'train'])
      plt.title(hp_name)
      plt.ylabel('rmsle')
      plt.xlabel('parameter value')
      if not png_path is None:
        plt.savefig(png_path)
      plt.show()
      
  if not disp_path is None:
    disp_to_file(result, disp_path)    
  
  return result