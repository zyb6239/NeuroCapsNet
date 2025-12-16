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

all_test_results = np.zeros([20,20])

if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')
if not os.path.exists('./my_results'):
    os.makedirs('./my_results')    


for sub in range(20):
    sub_path = '/disk2/ybzhang/dataset/asa_processed/' + 'Sub' + str(sub+1) + '.pkl'
    with open(sub_path, 'rb') as f:
        dataset = pickle.load(f)
        data = dataset['data']
        label = dataset['label']

    for tr in range(20):
        test_data = data[tr]
        test_label = label[tr]
        test_data = sliding_window(test_data,128,64)
        test_label = sliding_window(test_label,128,64)[:,0,:]

        train_data_trs = np.delete(data,tr,axis=0)
        train_label_trs = np.delete(label,tr,axis=0)
        train_data = []
        train_label = []
        for num in range(19):
            train_data.append(sliding_window(train_data_trs[num],128,64))
            train_label.append(sliding_window(train_label_trs[num],128,64)[:,0,:])
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)


        model = create_model(sample_len=sample_len, channels_num=channels_num, lr=lr)
        early_stopping = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'./checkpoint/Sub{sub+1}_{tr+1}_{model_name}.ckpt', monitor='loss', save_best_only=True)
        model.fit(train_data,train_label, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping, model_checkpoint])
        loss, all_test_results[sub,tr] = model.evaluate(test_data,test_label,batch_size=8)


df = pd.DataFrame(all_test_results, 
                index=[f'Subject_{i+1}' for i in range(20)],  # 设置行索引
                columns=[f'Trial_{i+1}' for i in range(20)])   # 设置列名

df.to_excel('./my_results/'+model_name+'.xlsx', sheet_name='Results', index=True)


