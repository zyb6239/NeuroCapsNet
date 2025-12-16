import os
import numpy as np
import pandas as pd
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from utils import sliding_window
from models.neurocap import create_model


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

model_name = 'neurocap'
sample_len = 128
channels_num = 64
lr = 5e-4
epochs = 100
batch_size = 64
num_class = 10

all_test_results = np.ones([20,10])

if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')
if not os.path.exists('./my_results'):
    os.makedirs('./my_results')    


for test_sub in range(20):
    traindata = []
    trainlabel = []

    for sub in range(20):
        test_id = str(sub+1)
        sub_path = '/disk2/ybzhang/dataset/asa_processed/' + 'Sub' + str(sub+1) + '.pkl'
        with open(sub_path, 'rb') as f:
            dataset = pickle.load(f)
            data = dataset['data']
            label = dataset['label']

        if sub == test_sub:
            testdata = data
            testlabel = label
        else:
            traindata.append(data)
            trainlabel.append(label)

    traindata = [sub_arr for sublist in traindata for sub_arr in sublist]
    trainlabel = [sub_arr for sublist in trainlabel for sub_arr in sublist]

    train_datas = []
    train_labels = []
    for train_tr in range(380):
        train_data = traindata[train_tr]
        train_label = trainlabel[train_tr]
        train_datas.append(sliding_window(train_data,sample_len,64))
        train_labels.append(sliding_window(train_label,sample_len,64)[:,0,:])

    train_datas = np.concatenate(train_datas,axis=0)
    train_labels = np.concatenate(train_labels,axis=0)

    model = create_model(sample_len=sample_len, channels_num=channels_num, lr=lr)
    early_stopping = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'./checkpoint/Sub{test_sub+1}_{model_name}.h5', monitor='loss', save_best_only=True,save_weights_only=True)
    model.fit(train_datas,train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    ft_trs = [0, 2, 4, 7, 9, 11, 12, 15, 17,19]
    ft_datas = []
    ft_labels = []
    for ft_tr in ft_trs:
        ft_data = testdata[ft_tr]
        ft_label = testlabel[ft_tr]
        ft_datas.append(sliding_window(ft_data,sample_len,64))
        ft_labels.append(sliding_window(ft_label,sample_len,64)[:,0,:])
    ft_datas = np.concatenate(ft_datas,axis=0)
    ft_labels = np.concatenate(ft_labels,axis=0)

    model.fit(ft_datas,ft_labels,epochs=10,batch_size=batch_size,shuffle=True,callbacks=model_checkpoint)
    
    num = 0
    for test_tr in range(20):
        if test_tr in ft_trs:
            continue
        test_data = testdata[test_tr]
        test_label = testlabel[test_tr]
        test_data = sliding_window(test_data,sample_len,64)
        test_label = sliding_window(test_label,sample_len,64)[:,0,:]

        all_test_results[test_sub,num] = model.evaluate(test_data,test_label,batch_size=8)[1]
        num = num + 1



df = pd.DataFrame(all_test_results, 
                index=[f'Subject_{i+1}' for i in range(20)],  
                columns=[f'Trial_{i+1}' for i in range(10)])   

df.to_excel('./my_results/'+model_name+'.xlsx', sheet_name='Results', index=True)


