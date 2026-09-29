import pickle
import pandas as pd
import numpy as np
import importlib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path
import argparse

'''
in each config file: 
model_dir: variable
load_data(lables):function
get_recent_model(): function
build_pipeline(): function
'''

Config_modules={
    'LogReg_RF_GB':"config_LR_RF_GB",
    'LogReg_RF_HEL':'config_LR_RF_HEL',
    'LogReg_RF_Spall':'config_LR_RF_Spall',
    'HGB_Spall':'config_HGB_Spall',
    'HGB_HEL':'config_HGB_HEL',
    'wvl_HEL':'config_wavelet_HEl',
    'HGB_GB':'config_HGB_GB',
    'FPCA_GB':'config_FPCA_GB',
    'FeatureEng_Spall':'config_FE_Spall',
    'FeatureEng_HEL':'config_FE_HEL',
    'labels_master':'labels.csv',
    'rL_HEL':'config_resLine_HEL',
    'weighted_rL_HEL':'config_wresL_HEL',
    'FE_spall_uncerts':'config_FE_Spall_uncerts',
    'GB_FE':'config_GB_FE'

}

def train_model(model_name):
    cfg=importlib.import_module(Config_modules[model_name])
    labels=pd.read_csv(Config_modules['labels_master'],names=['Filename','Classification'])
    labels['Good/Bad']=(~labels['Classification'].str.contains('6')).astype(int)
    labels['Vel_shot']=(labels['Classification'].str.contains('4')).astype(int)
    labels['Contains_Hel']=(labels['Classification'].str.contains('1|3')).astype(int)
    labels['Contains_Spall']=(labels['Classification'].str.contains('2|3')).astype(int)
    X, y = cfg.load_data(labels)
    pipeline_new=cfg.build_pipeline()
    #pipeline_new.fit(X,y)
    #if(os.path.exists(cfg.get_recent_model())):
        #with open(cfg.get_recent_model(), "rb") as f:
                #dict = pickle.load(f)
                #pipeline_old=dict['model']
        #compare_performance(pipeline_new,X, y,pipeline_old,cfg)
    #else:
    compare_performance(pipeline_new,X,y,cfg)

def compare_performance(pipeline_new,X,y,cfg,pipeline_old=None):
    print("results: updated model")
    Y_pred_new=cross_val_predict(pipeline_new,X,y,cv=5)
    class_report_new=classification_report(y,Y_pred_new,target_names=["Negative", "Positive"])
    print(class_report_new)
    cm_new=confusion_matrix(y,Y_pred_new,normalize='true')
    title=input("Confusion Matrix Title?").strip()
    ConfusionMatrixDisplay(confusion_matrix=cm_new,display_labels=['Negative', 'Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()
    if pipeline_old is not None: 
        print("results: previous model")
        Y_pred_old=pipeline_new.predict(pipeline_new,X,y,cv=5)
        class_report_old=classification_report(y,Y_pred_new,target_names=["Negative", "Positive"])
        print(class_report_old)
        cm=confusion_matrix(y,Y_pred_new,normalize='true')
        title=title +" previous"
        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Negative', 'Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
        plt.title(title)
        plt.savefig(f"{title}.png")
        plt.close()
    save_model=input("Save model? (y/n): ").strip().lower()
    if save_model in ["y", "yes"]:
         file_name=input("name: ").strip()
         pipeline_new.fit(X,y)
         pickle_name=f"{file_name}.pkl"
         path=Path(cfg.model_dir)/pickle_name
         path.parent.mkdir(parents=True,exist_ok=True)
         artifact={
              'model':pipeline_new,
              'cm':cm_new,
              'performance_training':class_report_new
         }
         with path.open("wb") as file:
            pickle.dump(artifact, file)

def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelName", type=str, required=True, help=f"Name of the model. Options: {', '.join(Config_modules.keys())}"
    )

    # 2. Parse incoming terminal arguments
    args = parser.parse_args()
    train_model(args.modelName)

    # 3. Use the arguments in your execution logic


if __name__ == "__main__":
    main()




    


         



         










    
    
    






