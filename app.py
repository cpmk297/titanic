import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

tab1, tab2, tab3= st.tabs(["Accueil","Analyse Exploratoire","Prédiction"])
with tab1:
    
    st.title("Le but de cette application est de prédire la survie ou non des passagers du titanic à l'aide le meilleur modèle(le SVM)")
    st.subheader('Auteur: Paul COFFI, Etudiant en data science')
    st.markdown("Après entrainement, le meilleur modèle est le Support Vector Machine(SVM) pour C=2.6 avec un noyau Gaussien." " " "C'est ce modèle qu'on va utiliser pour prédire la survie ou non des passagers")

with tab3:
    modele=joblib.load('model_svm.joblib')
    transformation=joblib.load('transformation.joblib')
    joblib.load('titanic_train.joblib')
    def predire(Pclass_, Sex_, Age_, Sibsp_, Parch_, Fare_, Embarked_):
       var= pd.DataFrame(
       {'Pclass': [Pclass_] , 'Sex': [Sex_], 'Age': [Age_], 'SibSp': [SibSp_], 'Parch':[Parch_], 'Fare': [Fare_], 'Embarked': [Embarked_] }, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
       if(modele.predict(var)==1):
           reponse= 'Le passager survit'
       else:
        reponse= 'Le passager ne survit pas'
       return reponse






    Pclass_= st.selectbox("Choisir la classe du passager", [1, 2, 3])
    Sex_= st.radio("Entrer le sexe du passager", ["male","female"])
    Age_= st.slider("Entrer l'age", 0,100 )
    SibSp_= st.slider('Entrer le nombre de frères et soeurs')  
    Parch_= st.selectbox('Entrer le nombre de parents', [0, 1, 2])
    Fare_= st.number_input('Entrer les frais de transport')
    Embarked_= st.selectbox('Entrer la gare de départ', joblib.load('titanic_train.joblib'))
    
    if st.button('Prédire'):
        prediction= predire(Pclass_, Sex_, Age_, SibSp_, Parch_, Fare_, Embarked_)
        st.success(prediction)
