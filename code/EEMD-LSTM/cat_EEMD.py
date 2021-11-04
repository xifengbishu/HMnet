import numpy as np
import sys

step = 15
batch = 256 
STA = int(sys.argv[1])
IMF = int(sys.argv[2])

EEMD=True
if IMF == 999:
	EEMD=False

def read_savez(train_model,nSTA,nIMF):
	result=np.load(str(train_model)+'_STA'+str(nSTA)+'_IMF'+str(nIMF)+'.npz')
	return result
'''
ww= read_savez('AR_LSTM',0,0)
print (ww['labels'])
print (ww['predict'])
'''

def evaluate_forecasts(y, y_hat):
	rmse = np.random.rand(len(y[0,:]))
	for i in range(len(rmse)):
		rmse[i] = np.sqrt(np.mean(np.square(y[:,i] - y_hat[:,i])))*0.01
		#print('t+%d RMSE: %f' % ((i+1), rmse[i]))
	return rmse

def model_rmse(train_mode):
	if EEMD :
		eemd_result = np.zeros((2,STA,IMF,batch,step))
		rmse = np.zeros((STA,step))
		for nsta in range(STA):
			for nimf in range(IMF):
				result=read_savez(train_mode,nsta,nimf)
				eemd_result[0,nsta,nimf,:,:]=result['predict']
				eemd_result[1,nsta,nimf,:,:]=result['labels']

		sla_result = np.sum(eemd_result,axis = 2)
		for nsta in range(STA):
			rmse[nsta,:] = evaluate_forecasts(sla_result[0,nsta,:,:],sla_result[1,nsta,:,:])
	else :
		eemd_result = np.zeros((2,STA,batch,step))
		rmse = np.zeros((STA,step))
		for nsta in range(STA):
			result=read_savez(train_mode,nsta,IMF)
			eemd_result[0,nsta,:,:]=result['predict']
			eemd_result[1,nsta,:,:]=result['labels']

		sla_result = eemd_result
		for nsta in range(STA):
			rmse[nsta,:] = evaluate_forecasts(sla_result[0,nsta,:,:],sla_result[1,nsta,:,:])

	return rmse

multi_performance = {}
multi_rmse = {}
eemd_multi_performance = {}
eemd_multi_rmse = {}


model = ['last_baseline','repeat_baseline','multi_linear_model','multi_dense_model','multi_conv_model','multi_lstm_model','AR_LSTM']
Lmodel = ['Last','Repeat','Linear','Dense','CNN','LSTM','AR_LSTM']

for n in range(len(model)):
	EEMD=True
	IMF = 5
	eemd_multi_performance[Lmodel[n]] = np.mean(model_rmse(model[n]))
	eemd_multi_rmse[Lmodel[n]] = model_rmse(model[n])

	EEMD=False
	IMF = 999
	multi_performance[Lmodel[n]] = np.mean(model_rmse(model[n]))
	multi_rmse[Lmodel[n]] = model_rmse(model[n])

print (" ----- ORI -----")
for name, value in multi_rmse.items():
  print (value)

print (" ----- EEMD -----")
for name, value in eemd_multi_rmse.items():
  print (value)

print (" ----- ORI -----")
for name, value in multi_performance.items():
  print(f'{name:8s}: {value:0.4f}')

print (" ----- EEMD -----")
for name, value in eemd_multi_performance.items():
  print(f'{name:8s}: {value:0.4f}')

