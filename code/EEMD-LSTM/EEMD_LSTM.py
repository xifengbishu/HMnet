# coding: utf-8
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')


import os
import sys
import datetime

import IPython
#import IPython.display
from IPython import get_ipython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import netCDF4 as nc

from PyEMD import EEMD
import numpy as np
#import pylab as pltt


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def evaluate_forecasts(y, y_hat):
	rmse = np.random.rand(len(y[0,:]))
	for i in range(len(rmse)):
		rmse[i] = np.sqrt(np.mean(np.square(y[:,i] - y_hat[:,i])))
		#print('t+%d RMSE: %f' % ((i+1), rmse[i]))
	return rmse

nSTA = int(sys.argv[1])
nIMF = int(sys.argv[2])

EEMD=True
if nIMF == 999:
	EEMD=False

OUT_STEPS = 15
input_width = 15 
label_width = 15

MAX_EPOCHS = 2
patience=50

# ========== set var =========
nsite   = 20        # number of SITESs
nimf  = 5         # nimf=0 , function will calculate

# ========== read =========
eemd_data = nc.Dataset('../../data/EEMD_LSTM.nc')     # 读取nc文件
eemd_dimensions, eemd_variables= eemd_data.dimensions, eemd_data.variables    # 获取文件中的维度和变量
# 29 var of NWP
osla	= eemd_variables['sla']
sla     = osla[:,:]*10000.0
eemd	= eemd_variables['eemd']


if EEMD :
	df_sla = pd.DataFrame(eemd[nSTA,nIMF,:])
	print ("EEMD")
else :
	df_sla = pd.DataFrame(sla[nSTA,:])
	print ("SLA")

df_sla.columns = ["sla"]
print (df_sla.head())

# This tutorial will just deal with **hourly predictions**, so start by sub-sampling the data from 10 minute intervals to 1h:

n = len(df_sla)
'''
#d_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
d_time = pd.date_range(start="19930101", end="20220101", freq="d") 
date_time = d_time[:n]
#sla_date_time = date_time[:n]

print (date_time[1900:1980:12])
print (date_time.shape)

# #### Time
# Similarly the `Date Time` column is very useful, but not in this string form. Start by converting it to seconds:
timestamp_s = date_time.map(pd.Timestamp.timestamp)
print (timestamp_s.shape)
'''

# We'll use a `(80%, 10%, 10%)` split for the training, validation, and test sets. Note the data is **not** being randomly shuffled before splitting. This is for two reasons.
# 
# 1. It ensures that chopping the data into windows of consecutive samples is still possible.
# 2. It ensures that the validation/test results are more realistic, being evaluated on data collected after the model was trained.

# In[16]:


column_indices = {name: i for i, name in enumerate(df_sla.columns)}

n = len(df_sla)
train_df = df_sla[0:int(n*0.8)]
val_df = df_sla[int(n*0.8):int(n*0.9)]
test_df = df_sla[int(n*0.9):]

num_features = df_sla.shape[1]

# ### Normalize the data

train_mean = train_df.mean()
train_std = train_df.std()

train_max = train_df.max()
train_min = train_df.min()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


multi_val_performance = {}
multi_performance = {}
multi_rmse = {}
# ## Data windowing
# 

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

# Here is code to create the 2 windows shown in the diagrams at the start of this section:

w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['sla'])
w1


# In[21]:


w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['sla'])
w2

# ### 2. Split

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# Try it out:

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

# Here is a plot method that allows a simple visualization of the split window:

w2.example = example_inputs, example_labels
# In[25]:
Pname = "w2"

# ### 4. Create `tf.data.Dataset`s

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=256,)

  ds = ds.map(self.split_window)

  return ds

def savez (train_model,predict,labels,rmse,train_std,nSTA,nIMF):
	predict = predict*train_std[0]*10000.
	labels = labels*train_std[0]*10000.
	np.savez(str(train_model)+'_STA'+str(nSTA)+'_IMF'+str(nIMF)+'.npz', predict=predict,labels=labels,rmse=rmse)

WindowGenerator.make_dataset = make_dataset


# The `WindowGenerator` object holds training, validation and test data. Add properties for accessing them as `tf.data.Datasets` using the above `make_dataset` method. Also add a standard example batch for easy access and plotting:

# In[29]:


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

w2.train.element_spec

# Iterating over a `Dataset` yields concrete batches:

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

wide_window = WindowGenerator(
    input_width=input_width, label_width=label_width, shift=1,
    label_columns=['sla'])

wide_window

def plot_history(history,modal_name):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    #plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    #plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.legend(loc='best',fontsize='small')  #set legend locationyy
    plt.title('loss')
    plt.savefig(str(modal_name)+'_loss_acc.jpg',dpi=300)
    plt.close()

def compile_and_fit(model, window, patience=patience):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  model.summary()
  return history


# ### Multi-output models

wide_window = WindowGenerator(
    input_width=input_width, label_width=label_width, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')


# ## Multi-step models

multi_window = WindowGenerator(input_width=input_width,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

repeat_multi_window = WindowGenerator(input_width=input_width,
                               label_width=input_width,
                               shift=input_width)
multi_window

# #### RNN

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    #tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    #tf.keras.layers.LSTM(128, return_sequences=False),
    #tf.keras.layers.LSTM(32, return_sequences=False),
    #tf.keras.layers.LSTM(32),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)
exit()
#IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=2)

print ('LSTM')
plot_history(history,'multi_lstm_model_STA'+str(nSTA)+'_IMF'+str(nIMF))
inputs, labels = multi_window.example
predict = multi_lstm_model.predict(inputs)

predict = np.reshape(predict, (predict.shape[0], predict.shape[1]))
labels  = np.reshape(labels, (labels.shape[0], labels.shape[1]))
rmse = evaluate_forecasts(labels,predict)
multi_rmse['LSTM'] = rmse

savez('multi_lstm_model',predict,labels,rmse,train_std,nSTA,nIMF)

print ('inputs.shape',inputs.shape)
print ('labels.shape',labels.shape)
print ('predict.shape',predict.shape)
exit()

# ### Advanced: Autoregressive model
# 
# The above models all predict the entire output sequence in a single step.
# 
# In some cases it may be helpful for the model to decompose this prediction into individual time steps. Then each model's output can be fed back into itself at each step and predictions can be made conditioned on the previous one, like in the classic [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).
# 
# One clear advantage to this style of model is that it can be set up to produce output with a varying length.
# 
# You could take any of the single-step multi-output models trained in the first half of this tutorial and run  in an autoregressive feedback loop, but here you'll focus on building a model that's been explicitly trained to do that.
# 
# ![Feedback a model's output to its input](images/multistep_autoregressive.png)
# 

# #### RNN
# 
# This tutorial only builds an autoregressive RNN model, but this pattern could be applied to any model that was designed to output a single timestep.
# 
# The model will have the same basic form as the single-step `LSTM` models: An `LSTM` followed by a `layers.Dense` that converts the `LSTM` outputs to model predictions.
# 
# A `layers.LSTM` is a `layers.LSTMCell` wrapped in the higher level `layers.RNN` that manages the state and sequence results for you (See [Keras RNNs](https://www.tensorflow.org/guide/keras/rnn) for details).
# 
# In this case the model has to manually manage the inputs for each step so it uses `layers.LSTMCell` directly for the lower level, single time step interface.

# In[84]:


class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)


# In[85]:


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)


# The first method this model needs is a `warmup` method to initialize its internal state based on the inputs. Once trained this state will capture the relevant parts of the input history. This is equivalent to the single-step `LSTM` model from earlier:

# In[86]:


def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup


# This method returns a single time-step prediction, and the internal state of the LSTM:

# In[87]:


prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape


# With the `RNN`'s state, and an initial prediction you can now continue iterating the model feeding the predictions at each step back as the input.
# 
# The simplest approach to collecting the output predictions is to use a python list, and `tf.stack` after the loop.

# Note: Stacking a python list like this only works with eager-execution, using `Model.compile(..., run_eagerly=True)` for training, or with a fixed length output. For a dynamic output length you would need to use a `tf.TensorArray` instead of a python list, and `tf.range` instead of the python `range`.

# In[88]:


def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)

  # Insert the first prediction
  predictions.append(prediction)

  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call


# Test run this model on the example inputs:

# In[89]:


print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)


# Now train the model:

# In[90]:


history = compile_and_fit(feedback_model, multi_window)

#IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=2)
multi_window.plot(feedback_model)
plt.savefig('multi_window_feedback_model.jpg',dpi=300)
plt.close()

print ('AR_LSTM')
feedback_model.summary()
plot_history(history,'feedback_model')
inputs, labels = multi_window.example
predict = feedback_model.predict(inputs)

predict = np.reshape(predict, (predict.shape[0], predict.shape[1]))
labels  = np.reshape(labels, (labels.shape[0], labels.shape[1]))
rmse = evaluate_forecasts(labels,predict)
multi_rmse['AR_LSTM'] = rmse

savez('AR_LSTM',predict,labels,rmse,train_std,nSTA,nIMF)

print (multi_rmse)

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')

for name, value in multi_rmse.items():
  print(f'{name:8s}: {value[1]:0.4f}')

f = open(r'multi_step_rmse.txt','w')
f.write(str(multi_rmse))

f = open(r'multi_step_performance2.txt','w')
f.write(str(multi_performance))

