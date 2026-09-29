import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#from Transformer_Classes import ResidualHELFeatureExtractor
#from Transformer_Classes import Spall_GlobalGeometryFeatureExtractor
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import os
import subprocess
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from datetime import datetime

Folder1=Path("Test_Data/08-20-2026")
def load_and_predict(shot=None, Folder=Folder1):
    Folder=Path(Folder)
    if shot:
        shots=[str(shot)]
    else:
        shots=[str(p) for p in Folder.rglob("*velocity--smooth.csv")]
    #print(len(shots))
    X_uNv=[]
    Y_pred_alg_HEL=[]
    Y_pred_alg_Spall=[]
    #print(len(shots))
    #shots1=[shots[1341]]
    for file in shots:
        temp=pd.read_csv(file,names=['t','v'])
        #plt.plot(temp['t'],temp['v'])
        #plt.savefig("testing.png")
        #plt.close()
        FileBase=file.partition("velocity--smooth.csv")[0]
        uncertFile=FileBase + "veluncert.csv"
        resultsFile=FileBase+"results.csv"
        uncerts=pd.read_csv(uncertFile, names=['t','u'])
        results=pd.read_csv(resultsFile)
        temp['SNR']=temp['v']/(uncerts['u']+1*10**(-6))
        both=[*temp['v'],*uncerts['u']]
        X_uNv.append(both)
        #print(type(results['HEL OK'][0]))
        #if type(results['HEL OK'][0]) is bool: 
            #return
        if results['HEL OK'][0]:
            Y_pred_alg_HEL.append(1)
        else:
            Y_pred_alg_HEL.append(0)
        if results['Spall OK'][0]:
            Y_pred_alg_Spall.append(1)
        else:
            Y_pred_alg_Spall.append(0)
        
    #print(type(X_uNv[0][0]))
    GB_detection=[]
    HEL_detection=[]
    Spall_detection=[]
    
    with open(Path(__file__).parent/".."/"Models"/"GB_FE_8_20_26.pkl",'rb') as file:
        GB_model=pickle.load(file)['model']
    with open(Path(__file__).parent/".."/"Models"/"weighted_rL_HEL_allData_8_20_26.pkl",'rb') as file:
        HEL_model=pickle.load(file)['model']
    with open(Path(__file__).parent/".."/"Models"/"FE_Spall_Uncerts_allData_8_20_26.pkl",'rb') as file:
        Spall_model=pickle.load(file)['model']
    for row in X_uNv:
        #print(len(row))
        GB_pred=GB_model.predict([row])
        GB_detection.append(GB_pred[0])
        #if GB_detection==1:
            #print('letting it predict')
        HEL_detection.append(HEL_model.predict([row])[0])
        Spall_detection.append(Spall_model.predict([row])[0])
        #else:
            #print('skipping!!')
            #HEL_detection.append(0)
            #Spall_detection.append(0)
    try:
        result=subprocess.run(f'attrib +U -P"{file}"',shell=True,check=True,capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"failed to offload file. Error: {e.stderr}")

    model_hel_corrected=[]
    model_spall_corrected=[]
    j=0
    for i in HEL_detection:
        if GB_detection[j]==0:
            model_hel_corrected.append(0)
            model_spall_corrected.append(0)
        else:
            model_hel_corrected.append(i)
            model_spall_corrected.append(Spall_detection[j])
        j=j+1

    artifact={
        "Shot":shots,
        "GB_results":GB_detection,
        "HEL_results":model_hel_corrected,
        "Spall_results":model_spall_corrected,
        "Alg_HEL_results":Y_pred_alg_HEL,
        "Alg_Spall_results":Y_pred_alg_Spall
    }
    date=datetime.now()
    if shot:
        pckl_name="results_"+str(shot.name)[:-4]+".pkl"
    else:
        pckl_name="results_"+str(Folder.name)+".pkl"
    path=Path(__file__).parent/Path("Results")/date.strftime("%Y-%d-%m")/pckl_name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact,file)
    print(path)
    return path

def generate_confusion_matrices(pickle_file,real_path,all=True):
    with open(pickle_file,'rb') as file:
        artifact=pickle.load(file)
    alg_hel=artifact['Alg_HEL_results']
    alg_spall=artifact['Alg_Spall_results']
    model_hel=artifact['HEL_results']
    print(len(model_hel))
    model_spall=artifact['Spall_results']
    GB_preds=artifact['GB_results']
    real=pd.read_csv(real_path,names=['filename','Classification'])
    real['HEL']=(real['Classification'].str.contains('1|3')).astype(int)
    alg_hel=alg_hel[:len(real['HEL'])]
    model_hel=model_hel[:len(real['HEL'])]
    real['Spall']=(real['Classification'].str.contains('2|3')).astype(int)
    alg_spall=alg_spall[:len(real['Spall'])]
    model_spall=model_spall[:len(real['Spall'])]
    real['GB']=(~real['Classification'].str.contains('6')).astype(int)
    model_GB=GB_preds[:len(real['GB'])]
    cm_GB=confusion_matrix(np.asarray(real['GB'],dtype=int),model_GB,normalize='true')
    class_report_mGB=classification_report(np.asarray(real['GB'],dtype=int),model_GB,target_names=["Negative", "Positive"])
    #print(class_report_mGB)
    
    cm_GB=confusion_matrix(np.asarray(real['GB'],dtype=int),model_GB)
    class_report_mGB=classification_report(np.asarray(real['GB'],dtype=int),model_GB,target_names=["Negative", "Positive"])
    ConfusionMatrixDisplay(cm_GB,display_labels=['Negative','Positive']).plot(cmap='Blues')
    plt.xlabel('GB Model')
    plt.ylabel('GB Real')
    date=datetime.now()
    base_path=Path(__file__).parent/Path("Results")/date.strftime("%Y-%d-%m")/str(Path(real_path).name)[:-4]
    fig_path=base_path/"cm_GB.png"
    fig_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    plt.close()
    
    cm_HEL=confusion_matrix(np.asarray(real['HEL'],dtype=int),model_hel,normalize='true')
    class_report_mHEL=classification_report(np.asarray(real['HEL'],dtype=int),model_hel,target_names=["Negative", "Positive"])
    cm_Spall=confusion_matrix(np.asarray(real['Spall'],dtype=int),model_spall,normalize='true')
    class_report_mSpall=classification_report(np.asarray(real['Spall'],dtype=int),model_spall,target_names=["Negative", "Positive"])
    cm_Hel_Alg=confusion_matrix(np.asarray(real['HEL'],dtype=int),alg_hel,normalize='true')
    class_report_aHEL=classification_report(np.asarray(real['HEL'],dtype=int),alg_hel,target_names=["Negative", "Positive"])
    cm_Spall_Alg=confusion_matrix(np.asarray(real['Spall'],dtype=int),alg_spall,normalize='true')
    class_report_aSpall=classification_report(np.asarray(real['HEL'],dtype=int),alg_spall,target_names=["Negative", "Positive"])

    ConfusionMatrixDisplay(cm_HEL,display_labels=['Negative','Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
    plt.title('Confusion Matrix HEL - Model')
    plt.xlabel('Model Predictions')
    plt.ylabel('True Classification')
    fig_path=base_path/"cm_mHEL.png"
    fig_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    plt.close()
    ConfusionMatrixDisplay(cm_Spall,display_labels=['Negative','Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
    plt.title('Confusion Matrix Spall - Model')
    plt.xlabel('Model Predictions')
    plt.ylabel('True Classification')
    fig_path=base_path/"cm_mSpall.png"
    fig_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    ConfusionMatrixDisplay(cm_Hel_Alg,display_labels=['Negative','Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
    plt.title('Confusion Matrix HEL - Alg')
    plt.xlabel('Algorithm Predictions')
    plt.ylabel('True Classification')
    fig_path=base_path/"cm_aHEL.png"
    fig_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    plt.close()
    ConfusionMatrixDisplay(cm_Spall_Alg,display_labels=['Negative','Positive']).plot(cmap='Blues',im_kw={'vmin': 0, 'vmax': 1})
    plt.title('Confusion Matrix Spall - Alg')
    plt.xlabel('Algorithm Predictions')
    plt.ylabel('True Classification')
    fig_path=base_path/"cm_aSpall.png"
    fig_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    print('model')
    print('GB')
    print(class_report_mGB)
    print('HEL')
    print(class_report_mHEL)
    print('Spall')
    print(class_report_mSpall)

    print('Algorithm')
    print('HEL')
    print(class_report_aHEL)
    print('Spall')
    print(class_report_aSpall)
    path=base_path/'class_reports.pkl'
    path.parent.mkdir(parents=True,exist_ok=True)
    class_reps={
        "Model_GB":class_report_mGB,
        "Model_HEL":class_report_mHEL,
        "Model_Spall":class_report_mSpall,
        "Alg_HEL":class_report_aHEL,
        "Alg_Spall":class_report_aSpall
    }
    with path.open("wb") as file:
        pickle.dump(class_reps,file)






def main():
    Folder=Path(__file__).parent/Path("Data/08-20-2026")
    shot=Path(__file__).parent/Path("Data/08-20-2026/C1--APLMAL00003_6a6cb96fd64c1367b86d16b3_2_2034_2026-08-12_17-57-43_shot01--00000-velocity--smooth.csv")
    pkl=load_and_predict(shot=None,Folder=Folder)
    #generate_confusion_matrices("Test/Results/2026-34-09/28/26/results.pkl","Test/Data/08-20-2026/test_real_answers.csv")

    # 3. Use the arguments in your execution logic


if __name__ == "__main__":
    main()

       
    





'''
with open('Spall/Feature_Eng/spall_feature_eng_8_13_26.pkl','rb') as file:
    loaded_data=pickle.load(file)

with open('Spall/HGB/rL_HEL_8_18_26.pkl','rb') as file:
    loaded_data_hel=pickle.load(file)
model=loaded_data['model']
model_HEL=loaded_data_hel['model']
test=pd.read_csv("alpss_output_data_set3\JHAMAL00004-03_2026-05-28_15-23-19_shot14_ch1-velocity--smooth.csv",names=['t','v'])['v'].to_numpy().reshape(1, -1)
plotting=pd.read_csv("alpss_output_data_set3\JHAMAL00004-03_2026-05-28_15-23-19_shot14_ch1-velocity--smooth.csv",names=['t','v'])
print(test.shape)
plt.plot(plotting['t'],plotting['v'])
plt.show()
plt.close()
print(model.predict(test))
print(model_HEL.predict(test))'''