import streamlit as st
import catboost as ctb
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
st.title('Method Of Repair Prediction for RC Columns')
algorithm= st.radio('ML algorithm:', ['CatBoost-Scenario VI','CatBoost-Scenario VII','Gradient Boost-Scenario X','MLP-Scenario VI','MLP-Scenario VII','MLP-Scenario X'])
url='https://drive.google.com/file/d/1CgUDJ-HGvdYrm0EYsykVMeZ0Ay0B04yo/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url)
data=df.head(422)
y = data["DS"]

def predictionCBVI (Aspect_Ratio, PF,HF):
 X1 = data[['H/B','pf', 'hf']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X1, y, test_size=0.3, random_state=1, stratify= y)
 model_CBC = ctb.CatBoostClassifier(depth = 5, iterations = 200, learning_rate = 0.1, border_count = 9, l2_leaf_reg = 10)
 model_CBC.fit(X_trainset, y_trainset)
 predicted_y = model_CBC.predict([Aspect_Ratio, PF,HF])
 return predicted_y

def predictionCBVII (Aspect_Ratio, PF,HF, R_1, R0, R1):
 X1 = data[['H/B','pf', 'hf','-1','0','1']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X1, y, test_size=0.3, random_state=1, stratify= y)
 model_CBC = ctb.CatBoostClassifier(depth = 5, iterations = 200, learning_rate = 0.1, border_count = 9, l2_leaf_reg = 10)
 model_CBC.fit(X_trainset, y_trainset)
 predicted_y = model_CBC.predict([Aspect_Ratio, PF,HF, R_1, R0, R1])
 return predicted_y

def predictionGB (A):
 X = data[['H/B', 'fc', 'Fyl', 'Fyv', 'ALr', 'AH', 'Av', 'S', 'Sc', 'hf', 'pf','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','10','9','8','7','6','5','4','3','2','1','0']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=1, stratify= y)
 GB = GradientBoostingClassifier(learning_rate=0.1, max_depth=9, n_estimators=250, subsample=0.4)
 GB.fit(X_trainset, y_trainset)
 predicted_y = GB.predict(A)
 return predicted_y

def predictionMLPVI (Aspect_Ratio, PF,HF):
 X1 = data[['H/B','pf', 'hf']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X1, y, test_size=0.3, random_state=1, stratify= y)
 mlp = MLPClassifier(hidden_layer_sizes=(6,6,4), max_iter= 1000,activation= 'tanh',solver= 'lbfgs',alpha= 0.1,learning_rate= 'constant',verbose=10)
 mlp.fit(X_trainset, y_trainset)
 predicted_y = mlp.predict([Aspect_Ratio, PF,HF])
 return predicted_y

def predictionMLPVII (Aspect_Ratio, PF,HF, R_1, R0, R1):
 X1 = data[['H/B','pf', 'hf','-1','0','1']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X1, y, test_size=0.3, random_state=1, stratify= y)
 mlp = MLPClassifier(hidden_layer_sizes=(5,5,8), max_iter= 1500,activation= 'tanh',solver= 'adam',alpha= 0.1,learning_rate= 'constant',verbose=10)
 mlp.fit(X_trainset, y_trainset)
 predicted_y = mlp.predict([Aspect_Ratio, PF,HF, R_1, R0, R1])
 return predicted_y

def predictionMLPX (A):
 X = data[['H/B', 'fc', 'Fyl', 'Fyv', 'ALr', 'AH', 'Av', 'S', 'Sc', 'hf', 'pf','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','10','9','8','7','6','5','4','3','2','1','0']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=1, stratify= y)
 mlp = MLPClassifier(hidden_layer_sizes=(9,14,28), max_iter= 1000,activation= 'tanh',solver= 'adam',alpha= 0.05,learning_rate= 'constant',verbose=10)
 mlp.fit(X_trainset, y_trainset)
 predicted_y = mlp.predict(A)
 return predicted_y


if algorithm=='CatBoost-Scenario VI':
      Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
      PF =st.number_input("Percolation feature")
      HF = st.number_input("Heterogeneity feature")
      yy=predictionCBVI(Aspect_Ratio, PF,HF)
elif algorithm=='CatBoost-Scenario VII':
         Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
         PF =st.number_input("Percolation feature")
         HF = st.number_input("Heterogeneity feature")
         R_1 = st.number_input("R_1")
         R0 = st.number_input("R0")
         R1 = st.number_input("R+1")
         yy=predictionCBVII(Aspect_Ratio, PF,HF, R_1, R0, R1)
elif algorithm=='Gradient Boost-Scenario X':
      Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
      fc =st.number_input("Compressive strength of concrete (MPa)")
      Fyl =st.number_input("Vertical rebar strength (MPa)")
      Fyv =st.number_input("Horizontal rebar strength (MPa)")
      AH =st.number_input("Longitudinal reinforcement ratio (%)")
      Av =st.number_input("Transverse reinforcement area")
      AL =st.number_input("Axial load capacity")
      S =st.number_input("Spacing of transverse reinforcement")
      Sc =st.number_input("Spacing of transverse reinforcement in critical area")
      PF = st.number_input("Percolation feature")
      HF = st.number_input("Heterogeneity feature")
      R_10 = st.number_input("R_10")
      R_9 = st.number_input("R_9")
      R_8 = st.number_input("R_8")
      R_7 = st.number_input("R_7")
      R_6 = st.number_input("R_6")
      R_5 = st.number_input("R_5")
      R_4 = st.number_input("R_4")
      R_3 = st.number_input("R_3")
      R_2 = st.number_input("R_2")
      R_1 = st.number_input("R_1")
      R0 = st.number_input("R0")
      R1 = st.number_input("R+1")
      R2 = st.number_input("R+2")
      R3 = st.number_input("R+3")
      R4 = st.number_input("R+4")
      R5 = st.number_input("R+5")
      R6 = st.number_input("R+6")
      R7 = st.number_input("R+7")
      R8 = st.number_input("R+8")
      R9 = st.number_input("R+9")
      R10 = st.number_input("R+10")
      a=np.asarray([[Aspect_Ratio, fc, Fyl, Fyv, AL, AH, Av, S, Sc, HF, PF,R_10,R_9,R_8,R_7,R_6,R_5,R_4,R_3,R_2,R_1,R10,R9,R8,R7,R6,R5,R4,R3,R2,R1,R0]])
      yy=predictionGB(a)
elif algorithm=='MLP-Scenario VI':
     Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
     PF =st.number_input("Percolation feature")
     HF = st.number_input("Heterogeneity feature")
     yy=predictionMLPVI(np.array([Aspect_Ratio, PF,HF]).reshape(1,-1))
elif algorithm=='MLP-Scenario VII':
         Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
         PF =st.number_input("Percolation feature")
         HF = st.number_input("Heterogeneity feature")
         R_1 = st.number_input("R_1")
         R0 = st.number_input("R0")
         R1 = st.number_input("R+1")
         yy=predictionMLPVII(Aspect_Ratio, PF,HF, R_1, R0, R1)
else: 
      Aspect_Ratio =st.number_input("Aspect Ratio__H/B")
      fc =st.number_input("Compressive strength of concrete (MPa)")
      Fyl =st.number_input("Vertical rebar strength (MPa)")
      Fyv =st.number_input("Horizontal rebar strength (MPa)")
      AH =st.number_input("Longitudinal reinforcement ratio (%)")
      Av =st.number_input("Transverse reinforcement area")
      AL =st.number_input("Axial load capacity")
      S =st.number_input("Spacing of transverse reinforcement")
      Sc =st.number_input("Spacing of transverse reinforcement in critical area")
      PF = st.number_input("Percolation feature")
      HF = st.number_input("Heterogeneity feature")
      R_10 = st.number_input("R_10")
      R_9 = st.number_input("R_9")
      R_8 = st.number_input("R_8")
      R_7 = st.number_input("R_7")
      R_6 = st.number_input("R_6")
      R_5 = st.number_input("R_5")
      R_4 = st.number_input("R_4")
      R_3 = st.number_input("R_3")
      R_2 = st.number_input("R_2")
      R_1 = st.number_input("R_1")
      R0 = st.number_input("R0")
      R1 = st.number_input("R+1")
      R2 = st.number_input("R+2")
      R3 = st.number_input("R+3")
      R4 = st.number_input("R+4")
      R5 = st.number_input("R+5")
      R6 = st.number_input("R+6")
      R7 = st.number_input("R+7")
      R8 = st.number_input("R+8")
      R9 = st.number_input("R+9")
      R10 = st.number_input("R+10")
      a=np.asarray([[Aspect_Ratio, fc, Fyl, Fyv, AL, AH, Av, S, Sc, HF, PF,R_10,R_9,R_8,R_7,R_6,R_5,R_4,R_3,R_2,R_1,R10,R9,R8,R7,R6,R5,R4,R3,R2,R1,R0]])
      yy=predictionMLPX(a)


st.button("DS prediction")
st.write(yy)
