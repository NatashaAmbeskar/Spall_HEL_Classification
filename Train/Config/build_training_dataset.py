import pandas as pd
import numpy as np

def build_training_dataset_gb(labels,needed_data,uniform_length=False):
    X_gb_SNR=[]
    X_gb=[]
    X_uNv_gb=[]
    for file in labels['Filename']:
        temp=pd.read_csv(file,names=['t','v'])
        uncertFile=file.partition("velocity--smooth.csv")[0]
        uncertFile=uncertFile + "veluncert.csv"
        uncerts=pd.read_csv(uncertFile, names=['t','u'])
        temp['SNR']=temp['v']/(uncerts['u']+1*10**(-6))
        X_gb_SNR.append(temp['SNR'].tolist())
        X_gb.append(temp['v'].tolist())
        both=[*temp['v'],*uncerts['u']]
        X_uNv_gb.append(both)
    Y_gb=labels['Good/Bad'].tolist()
    if uniform_length:
        clean_data = [(x, y, z, w) for x, y, z, w in zip(X_gb, Y_gb, X_gb_SNR,X_uNv_gb) if len(x) == 4000]

        X_gb_SNR=np.array([item[2] for item in clean_data])
        X_gb=np.array([item[0] for item in clean_data])
        Y_gb=np.array([item[1] for item in clean_data])
        X_uNv_gb=np.array([item[3] for item in clean_data])
    if needed_data == "unv":
        return X_uNv_gb, Y_gb
    if needed_data=="uncerts":
        return X_gb_SNR, Y_gb
    else:
        return X_gb, Y_gb


def build_training_dataset_hel(labels, needed_data, uniform_length=False):
    labels_good=labels[(labels['Good/Bad']==1) & (labels['Vel_shot']==0)].copy()
    X_hel_SNR=[]
    X_vels_hS=[]
    X_uNv_hS=[]
    X_uncerts_hS=[]
    for file in labels_good['Filename']:
        temp=pd.read_csv(file,names=['t','v'])
        uncertFile=file.partition("velocity--smooth.csv")[0]
        uncertFile=uncertFile + "veluncert.csv"
        uncerts=pd.read_csv(uncertFile, names=['t','u'])
        temp['SNR']=temp['v']/(uncerts['u']+1*10**(-6))
        X_hel_SNR.append(temp['SNR'].tolist())
        X_vels_hS.append(temp['v'].tolist())
        X_uncerts_hS.append(uncerts['u'].tolist())
        both=[*temp['v'],*uncerts['u']]
        X_uNv_hS.append(both)
    Y_hel=labels_good['Contains_Hel'].tolist()
    if uniform_length:
        clean_data = [(x, y, z) for x, y, z in zip(X_vels_hS, X_uNv_hS, Y_hel) if len(x) == 4000]

        X_vels_hS=np.array([item[0] for item in clean_data])
        X_uNv_hS=np.array([item[1] for item in clean_data])
        Y_hel=np.array([item[2] for item in clean_data])
    if needed_data == "unv":
        return X_uNv_hS, Y_hel
    else:
        return X_vels_hS, Y_hel

def build_training_dataset_spall(labels,needed_data, uniform_length=False):
    labels_good=labels[(labels['Good/Bad']==1) & (labels['Vel_shot']==0)].copy()
    X_spall_SNR=[]
    X_vels_hS=[]
    X_uNv_hS=[]
    X_uncerts_hS=[]
    for file in labels_good['Filename']:
        temp=pd.read_csv(file,names=['t','v'])
        uncertFile=file.partition("velocity--smooth.csv")[0]
        uncertFile=uncertFile + "veluncert.csv"
        uncerts=pd.read_csv(uncertFile, names=['t','u'])
        temp['SNR']=temp['v']/(uncerts['u']+1*10**(-6))
        X_spall_SNR.append(temp['SNR'].tolist())
        X_vels_hS.append(temp['v'].tolist())
        X_uncerts_hS.append(uncerts['u'].tolist())
        both=[*temp['v'],*uncerts['u']]
        X_uNv_hS.append(both)
    Y_spall=labels_good['Contains_Spall'].tolist()
    if uniform_length:
        clean_data = [(x, y, z) for x, y, z in zip(X_vels_hS, X_uNv_hS,Y_spall) if len(x) == 4000]
        X_vels_hS=np.array([item[0] for item in clean_data])
        X_uNv_hS=np.array([item[1] for item in clean_data])
        Y_spall=np.array([item[2] for item in clean_data])
    if needed_data == "unv":
        return X_uNv_hS, Y_spall
    else:
        return X_vels_hS, Y_spall

    
