# -*- coding: utf-8 -*-
"""
Spyder Editor

@Author  Siddhartha Sarkar
"""
    
 ###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import seaborn as sns
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn import preprocessing, decomposition, model_selection
from sklearn.model_selection import cross_val_score
#for visualization
import cufflinks as cf
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
##pyLDAvis.enable_notebook()
import plotly.express as px
#for modelimplementation
import pickle
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
le_encoder=LabelEncoder()
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
#init_notebook_mode(connected = True)
import plotly.figure_factory as ff

sns.set_palette("hls")
plt.style.use('fivethirtyeight')

# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report


# Custome Component Fxn
import sweetviz as sv 
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#lottie animations
import time
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

le_encoder=LabelEncoder()
###############################################Data Processing###########################


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
project_url_1="https://assets9.lottiefiles.com/packages/lf20_bzgbs6lx.json"
project_url_2="https://assets6.lottiefiles.com/packages/lf20_eeuhulsy.json"
report_url="https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json"
about_url="https://assets2.lottiefiles.com/packages/lf20_k86wxpgr.json"

about_1=load_lottieurl(about_url)
report_1=load_lottieurl(report_url)
project_1=load_lottieurl(project_url_1)
project_2=load_lottieurl(project_url_2)

lottie_download = load_lottieurl(lottie_url_download)

#st_lottie(lottie_hello, key="hello")


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


final_data=pd.read_csv("final_cleaned_data.csv")
final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
final_data["AQI_Bucket"]=le_encoder.fit_transform(final_data["AQI_Bucket"])



#final_data["City"]=le_encoder.fit_transform(final_data["City"])
def user_input_features():
    
    PM25 = st.sidebar.number_input("Insert PM2.5 value")
    PM10 = st.sidebar.number_input("Insert PM10 value")
    NO = st.sidebar.number_input("Insert NO value")
    NO2= st.sidebar.number_input("Insert NO2  value")
    NOx = st.sidebar.number_input("Insert NOx value")
    NH3 = st.sidebar.number_input("Insert NH3 value")
    CO = st.sidebar.number_input("Insert CO value")
    SO2 = st.sidebar.number_input("Insert SO2 value")
    O3 = st.sidebar.number_input("Insert O3 value")
    Benzene = st.sidebar.number_input("Insert Benzene  value")
    Toluene = st.sidebar.number_input("Insert Toluene value")
    Xylene = st.sidebar.number_input("Insert Xylene value")
    City   = st.sidebar.selectbox("City", final_data["City"].unique())
    data = {'PM2.5':PM25,
            'PM10':PM10,
            'NO':NO,
            'NO2':NO2,
            'NOx':NOx,
            'NH3':NH3,
            'CO':CO,
            'SO2':SO2,
            'O3':O3,
            'Benzene':Benzene,
            'Toluene':Toluene,
            'Xylene':Xylene,
            'City':City,
            }
    features = pd.DataFrame(data,index = [0])
    return features 




###############################################Exploratory Data Analysis###############################################

#For Label Analysis
def label_analysis():
    
    def plot1():
        final_data=pd.read_csv("final_cleaned_data.csv")
        explode=[0.2,0.1,0.05,0.04,0.07,0.06,0.1,0.06,0.2,0.06,
             0.06,0.06,0.06,0.06,0.03,0.06,0.02,0.01,0.06,0.06,
             0.02,0.01,0.06,0.03,0.06,0.02]
        plt.figure(figsize=(17,8))
        plt.pie(x=final_data.City.value_counts(),explode=explode,labels=final_data.City.unique(),
            colors=None,autopct="%1.2f%%",shadow=True,wedgeprops=None,textprops=None,
        )
        plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0)
        plt.show()
    p1=plot1()
    st.write("Cities Of India")
    st.pyplot(p1)
    
    def plot2():
        final_data=pd.read_csv("final_cleaned_data.csv")
        label =final_data.City.unique()
        val = final_data.City.value_counts()
        #percent = [15.71, 15.58, 15.34, 14.72, 14.48, 13.13, 11.04]
        plt.figure(figsize=(17,8))
        data1 = {
       "values": val,
       "labels": label,
       "domain": {"column": 0},
       "name":"City" ,
       "hoverinfo":"label",
       "hole": .2,
       "type": "pie"
            }
        data = [data1]
        layout = go.Layout(
        {
          "title":"Cities",
          "grid": {"rows": 1, "columns": 1},
          "annotations": [
             {
                "font": {
                   "size": 20
                },
                "showarrow": False,
                "text": "City",
                "x": 0.50,
                "y": 0.5
             }]})
          


        fig = go.Figure(data = data, layout = layout)
        return fig
        #iplot(fig)
    
    
    p2=plot2()
    st.write("Cities Of India")
    st.plotly_chart(p2)
    def plot3():
        final_data=pd.read_csv("final_cleaned_data.csv")
        #plt.figure(figsize=(15,8))
        #labels=final_data.AQI_Bucket.unique()
        #sns.barplot(x=final_data.AQI_Bucket.unique(),
               #y=final_data.AQI_Bucket.value_counts(),data=final_data,palette='tab10')
        #plt.xticks(rotation=90)
        #plt.legend(labels)
        #plt.show()
        
        
        explode=[0.07,0.01,0.05,0.04,0.07,0.06]
        plt.figure(figsize=(17,8))
        plt.pie(x=final_data.AQI_Bucket.value_counts(),explode=explode,labels=final_data.AQI_Bucket.unique(),
            colors=None,autopct="%1.2f%%",shadow=True,wedgeprops=None,textprops=None,
             )
        plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0)
        plt.show()
    
    
    p3=plot3()
    st.write("Air Qualities Of Indian Cities")
    st.pyplot(p3)
    
    def plot4():
        final_data=pd.read_csv("final_cleaned_data.csv")
        import plotly.graph_objs as go
        label =final_data.AQI_Bucket.unique()
        val = final_data.AQI_Bucket.value_counts()
        #percent = [15.71, 15.58, 15.34, 14.72, 14.48, 13.13, 11.04]
        import plotly.graph_objs as go
        plt.figure(figsize=(17,8))
        data1 = {
       "values": val,
       "labels": label,
       "domain": {"column": 0},
       "name":"AQI_Modes" ,
       "hoverinfo":"label",
       "hole": .2,
       "type": "pie"
         }
        data = [data1]
        layout = go.Layout(
          {
          "title":"Different Types of Pollutants in the Air",
          "grid": {"rows": 1, "columns": 1},
          "annotations": [
             {
                "font": {
                   "size": 20
                },
                "showarrow": False,
                "text": "AQI_Modes",
                "x": 0.50,
                "y": 0.5
             }]})
          


        fig = go.Figure(data = data, layout = layout)
        return fig
        #iplot(fig)
    p4=plot4()
    st.write(" Different kinds Of Pollutants in the Air")
    st.plotly_chart(p4)
    
    def plot5():
        final_data=pd.read_csv("final_cleaned_data.csv")
        final_data.groupby(by="AQI_Bucket")["PM2.5","PM10","NO",'NO2','NOx','NH3','CO','SO2',"O3","Benzene",'Toluene','Xylene'].agg('mean').plot(
           kind="bar",figsize=(15,7),grid=True)
        
    p5=plot5()
    st.write(" Different kinds Of Pollutants in the Air")
    st.pyplot(p5)    
    
    
    def plot7():
        final_data=pd.read_csv("final_cleaned_data.csv")
        final_data.groupby(by="City")["PM2.5","PM10","NO",'NO2','NOx','NH3','CO','SO2',"O3","Benzene",'Toluene','Xylene'].agg('mean').plot(
                kind="bar",grid=True,figsize=(17,7))
        
    p7=plot7()
    st.write(" Different kinds Of Pollutants in the Cities")
    st.pyplot(p7)  
    
    
    fig=Image.open("index.png")
    st.image(fig)
    
    col1,col2=st.columns(2)
    col3,col4=st.columns(2)
    with col1:
        fig1=Image.open("index4.png")
        col1.image(fig1)
    with col2:
        fig2= Image.open("index1.png")
        col2.image(fig2)
        
    with col3:
        fig3= Image.open("index2.png")
        col3.image(fig3)
        
    with col4:
        
       fig4= Image.open("index3.png")
       col4.image(fig4)
        
    
    
    
    
        
        
        

    
    
    
    
    
    def plot6():
        plt.figure(figsize=(12,6))
        sns.distplot(final_data["PM2.5"],hist=False,color="green",label="PM2.5")
        sns.distplot(final_data["PM10"],hist=False,color="blue",label="PM10")
        sns.distplot(final_data["NO"],hist=False,color="pink",label="NO")
        sns.distplot(final_data["NO2"],hist=False,color="yellow",label="NO2")
        sns.distplot(final_data["NOx"],hist=False,color="violet",label="NOx")
        sns.distplot(final_data["NH3"],hist=False,color="m",label="NH3")
        sns.distplot(final_data["CO"],hist=False,color="coral",label="CO")
        sns.distplot(final_data["SO2"],hist=False,color="cyan",label="SO2")
        sns.distplot(final_data["O3"],hist=False,color="magenta",label="O3")
        sns.distplot(final_data["Benzene"],hist=False,color="teal",label="Benzene")
        sns.distplot(final_data["Toluene"],hist=False,color="lavender",label="Toluene")
        sns.distplot(final_data["Xylene"],hist=False,color="olive",label="Xylene")
        sns.distplot(final_data["O3"],hist=False,color="magenta",label="O3")
        plt.ylim(0,0.10)
        plt.xlim(0,200)
        plt.xlabel("Pollutants")
        plt.yscale("symlog")
        plt.legend()
        plt.show()
    
    p6=plot6()
    st.write(" Different kinds Of Pollutants in the Air")
    st.pyplot(p6)
    st.sidebar.header('Settings')

    actions = {'Pie_Plot1': p1, 'Pie_Plot2': p2,'Pie_Plot3': p3,'Pie_Plot4': p4,'Bar_Plot': p5,'Dist_Plot': p6}
    choices = st.sidebar.multiselect('Choose task:', ['Pie_Plot1','Pie_Plot2','Pie_Plot3','Pie_Plot4', 'Bar_Plot','Dist_Plot'])
    for choice in choices:
        result = actions[choice]()
    
def get_data_class(final_data):
    final_data=pd.read_csv("final_cleaned_data.csv")
    final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
    final_data["AQI_Bucket"]=le_encoder.fit_transform(final_data["AQI_Bucket"])
    final_data["City"]=le_encoder.fit_transform(final_data["City"])
    X1=pd.read_csv("cleaned_final_data11.csv")
    y1=final_data["AQI_Bucket"]
    X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=42)
    
    return X1_train,X1_test,y1_train,y1_test
  


def get_data_reg(final_data):
    final_data=pd.read_csv("final_cleaned_data.csv")
    final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
    final_data["AQI_Bucket"]=le_encoder.fit_transform(final_data["AQI_Bucket"])
    final_data["City"]=le_encoder.fit_transform(final_data["City"])
    X=pd.read_csv("cleaned_final_data11.csv")
    y=final_data["AQI"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    
    return X_train,X_test,y_train,y_test  
    

############################################### Model Learning ###############################################

#multi-class log-loss
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota





#Model Logistic Regression
def logistic_regression(final_data):
    
    X1_train,X1_test,y1_train,y1_test=get_data_class(final_data)
    lr= LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    lr.fit(X1_train, y1_train)
    
    #predict y value for dataset
    y_predict= lr.predict(X1_test)
    y_prob= lr.predict_proba(X1_test)
    
    report= classification_report(y1_test,y_predict, output_dict=True)
    tab=pd.DataFrame(report).transpose()
    st.write(tab)
   
    
    logloss=multiclass_logloss(y1_test, y_prob)
    st.write("logloss: %0.3f " % logloss) 
    
    
    st.subheader("Confusion Matrix") 
    conf_matrix= confusion_matrix(y1_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='rainbow',fmt=".1f")
    st.pyplot()

    
#Model XGB Classifier
def xgb_classifier(final_data):
    X1_train,X1_test,y1_train,y1_test=get_data_class(final_data)
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(X1_train, y1_train)
    y_predict = clf.predict(X1_test)
    y_prob= clf.predict_proba(X1_test)
    
    report=classification_report(y1_test,y_predict, output_dict=True)
    tab2=pd.DataFrame(report).transpose()
    st.write(tab2)
    
    logloss=multiclass_logloss(y1_test, y_prob)
    st.write("logloss: %0.3f " % logloss) 
    
   
    
    st.subheader("Confusion Matrix") 
    conf_matrix= confusion_matrix(y1_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='rainbow',fmt=".1f")
    st.pyplot()
    
    

    

    
    
#Model Random Forest Classifier
def randomforest_classifier(final_data):
    X1_train,X1_test,y1_train,y1_test=get_data_class(final_data)
    clf= RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,
                                max_features='auto',random_state=None,class_weight="balanced")
    clf.fit(X1_train, y1_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X1_test)
    y_prob= clf.predict_proba(X1_test)
    
    report=classification_report(y1_test,y_predict, output_dict=True)
    tab5=pd.DataFrame(report).transpose()
    st.write(tab5)
    
    logloss=multiclass_logloss(y1_test, y_prob)
    st.write("logloss: %0.3f " % logloss)
    
    
        
    st.subheader("Confusion Matrix")    
    conf_matrix= confusion_matrix(y1_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='rainbow',fmt=".1f")
    st.pyplot()

    
#Model Random Forest Regressor
def randomforest_regressor(final_data):
    X_train,X_test,y_train,y_test=get_data_reg(final_data)
    rf_reg=RandomForestRegressor(n_estimators=100,criterion='squared_error',max_depth=8,
                                 max_features='auto')
    rf_reg.fit(X_train,y_train)
    y_train_pred=rf_reg.predict(X_train)
    y_test_pred=rf_reg.predict(X_test)
    r2_score(y_train,y_train_pred)
    
    r2_score(y_test,y_test_pred)
    
    m1=mean_absolute_error(y_train,y_train_pred)
    st.write("Mean Absolute error: %0.3f " % m1)

    m2=mean_squared_error(y_train,y_train_pred)
    st.write("Mean squared error: %0.3f " % m2)
    m3=mean_absolute_percentage_error(y_train,y_train_pred)

#Model XGBoost  Regressor
def XGboost_regressor(final_data):
    X_train,X_test,y_train,y_test=get_data_reg(final_data)
    xgb_reg=XGBRegressor(n_estimators=150,max_depth=8,learning_rate=0.05,booster="gbtree")
    xgb_reg.fit(X_train,y_train)
    y_train_pred=xgb_reg.predict(X_train)
    y_test_pred=xgb_reg.predict(X_test)
    r2_score(y_train,y_train_pred)
    
    r2_score(y_test,y_test_pred)
    
    m1=mean_absolute_error(y_train,y_train_pred)
    st.write("Mean Absolute error: %0.3f " % m1)

    m2=mean_squared_error(y_train,y_train_pred)
    st.write("Mean squared error: %0.3f " % m2)
    m3=mean_absolute_percentage_error(y_train,y_train_pred)

def predict_func(df):
    df["City"]=le_encoder.fit_transform(df["City"])
    final_data=pd.read_csv("final_cleaned_data.csv")
    final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
    X1_train,X1_test,y1_train,y1_test=get_data_class(final_data)
    xgb_classifer= XGBClassifier(n_estimators=200,max_depth=8,booster="gbtree",learning_rate=0.005,class_weight={1:1,2:4,3:1.2,4:5.4,5:4.66,0:7.5})
    xgb_classifer.fit(X1_train,y1_train)
    xgb_train_predict=xgb_classifer.predict(X1_train)
    xgb_prediction = xgb_classifer.predict(df)
    
    return  xgb_prediction

def predict_func1(df):
    df["City"]=le_encoder.fit_transform(df["City"])
    final_data=pd.read_csv("final_cleaned_data.csv")
    final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
    X_train,X_test,y_train,y_test=get_data_reg(final_data)
    xgb_reg=XGBRegressor(n_estimators=150,max_depth=8,learning_rate=0.05,booster="gbtree")
    xgb_reg.fit(X_train,y_train)
    y_train_pred=xgb_reg.predict(X_train)
    y_test_pred=xgb_reg.predict(df)
    #xgb_prediction = xgb_classifer.predict(df)
    return y_test_pred
###############################################Streamlit Main###############################################

def main():
    # set page title
    
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title=None, options=["Home", "Projects","Report" ,"About"], icons=["house", "book","app-indicator","envelope"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": " #f08080 "},"icon": {"color": "blue", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#eee", },           "nav-link-selected": {"background-color": "green"},},)
    
    #horizontal Home selected
    if selected == "Home":
        home_url= "https://assets5.lottiefiles.com/packages/lf20_ygiuluqn.json"
        home12=load_lottieurl(home_url)
        st_lottie(home12,key= "key1234")
        #image= Image.open("airquality.png")
        #st.image(image,use_column_width=True)
        
        with st.sidebar:
            st.title("Home")
            st_lottie(lottie_hello, key="hello")
            st.write('Author@ Siddhartha Sarkar')
            st.write('Data Scientist ')
        st.balloons()
# =============================================================================
#             image= Image.open("Home.png")
#             add_image=st.image(image,use_column_width=True)  
#         st.sidebar.markdown("[ Visit To Github Repositories](.git)")    
# =============================================================================
        
        #st.title('Air Quality Index')
        #st.video("https://youtu.be/O73OPzkUlR0")
        
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> This Project Aims to Predict the  Quality of Air and also Measures the AQI(Air Quality Index)</h1>
		</div>  """
        
		
        components.html(html_temp)
        st.subheader("Measurment of Air Quality Index of Several Indian Cities")
        
        st.subheader("Air Quality Index:")
        def header(url):
            st.markdown(f'<p style="background-color:#87CEFA ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
                       An air quality index (AQI) is used by government agencies[1] to communicate to the public how 
                       polluted the air currently is or how polluted it is forecast to become.[2][3] AQI information 
                       is obtained by averaging readings from an air quality sensor, which can increase due to vehicle 
                       traffic, forest fires, or anything that can increase air pollution. Pollutants tested include ozone, 
                       nitrogen dioxide, sulphur dioxide, among others.
                       Public health risks increase as the AQI rises, especially affecting children, the elderly, and 
                       individuals with respiratory or cardiovascular issues. During these times, governmental bodies 
                       generally encourage people to reduce physical activity outdoors, or even avoid going out altogether. 
                       The use of face masks such as cloth masks may also be recommended.
                       Different countries have their own air quality indices, corresponding to different national air 
                       quality standards. Some of these are the Air Quality Health Index (Canada), the 
                       Air Pollution Index (Malaysia), and the Pollutant Standards Index (Singapore). <br>
                       Computation of the AQI requires an air pollutant concentration over a specified averaging period, 
                       obtained from an air monitor or model. Taken together, concentration and time represent the dose of 
                       the air pollutant. Health effects corresponding to a given dose are established by epidemiological 
                       research.[4] Air pollutants vary in potency, and the function used to convert from air pollutant 
                       concentration to AQI varies by pollutant. Its air quality index values are typically grouped into 
                       ranges. Each range is assigned a descriptor, a color code, and a standardized public health advisory.
                        The AQI can increase due to an increase of air emissions. For example, during rush hour traffic 
                        or when there is an upwind forest fire or from a lack of dilution of air pollutants. Stagnant air,
                        often caused by an anticyclone, temperature inversion, or low wind speeds lets air pollution remain
                        in a local area, leading to high concentrations of pollutants, chemical reactions between air 
                        contaminants and hazy conditions.
                        Signboard in Gulfton, Houston indicating an ozone watch
                        On a day when the AQI is predicted to be elevated due to fine particle pollution, 
                        an agency or public health organization might:
                        advise sensitive groups, such as the elderly, children, and those with respiratory or 
                        cardiovascular problems, to avoid outdoor exertion.
                        declare an "action day" to encourage voluntary measures to reduce air emissions, such as using 
                        public transportation.
                        recommend the use of masks to keep fine particles from entering the lungs
                        During a period of very poor air quality, such as an air pollution episode,
                        when the AQI indicates that acute exposure may cause significant harm to the public health, 
                        agencies may invoke emergency plans that allow them to order major 
                        emitters (such as coal burning industries) to curtail emissions until the hazardous 
                        conditions abate.
                        Most air contaminants do not have an associated AQI. Many countries monitor ground-level ozone, 
                        particulates, sulfur dioxide, carbon monoxide and nitrogen dioxide, and calculate air quality
                        indices for these pollutants.
                        he definition of the AQI in a particular nation reflects the discourse surrounding the 
                        development of national air quality standards in that nation.[11] A website allowing 
                        government agencies anywhere in the world to submit their real-time air monitoring data 
                        for display using a common definition of the air quality index has recently become available
                                           
        
		  """
        
		
        header(html_temp11)
        st.header("National Air Quality Index (AQI) For India")
        def header(url):
            st.markdown(f'<p style="background-color:#87CEFA ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
		               The Central Pollution Control Board along with State Pollution Control Boards has been operating 
                       National Air Monitoring Program (NAMP) covering 240 cities of the country having more than 342 
                       monitoring stations. An Expert Group comprising medical professionals, air quality experts, 
                       academia, advocacy groups, and SPCBs was constituted and a technical study was awarded to IIT Kanpur
                       . IIT Kanpur and the Expert Group recommended an AQI scheme in 2014. While the earlier measuring
                       index was limited to three indicators, the new index measures eight parameters. 
                       The continuous monitoring systems that provide data on near real-time basis are installed 
                       in New Delhi, Mumbai, Pune, Kolkata and Ahmedabad.
                       There are six AQI categories, namely Good, Satisfactory, Moderate, Poor, Severe, and Hazardous. 
                       The proposed AQI will consider eight pollutants (PM10, PM2.5, NO2, SO2, CO, O3, NH3, and Pb) for 
                       which short-term (up to 24-hourly averaging period) National Ambient Air Quality Standards are prescribed. 
                       Based on the measured ambient concentrations, corresponding standards and likely health impact, a sub-index is 
                       calculated for each of these pollutants. The worst sub-index reflects overall AQI. Likely health impacts for different
                       AQI categories and pollutants have also been suggested, with primary inputs from the medical experts in the group
		  """
        
		
        header(html_temp11)
        st.subheader("AQI Category, Pollutants and Health Breakpoints ")
        image= Image.open("img125.png")
        st.image(image,use_column_width=True)
        st.subheader("Associated Health Impacts  ")
        image= Image.open("img1254.png")
        st.image(image,use_column_width=True)
        
   
        ### features
        image= Image.open("img.jpg")
        st.sidebar.image(image,use_column_width=True)
        
        
        def plot12():
            final_data=pd.read_csv("final_cleaned_data.csv")
            final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
            final_data["AQI_Bucket"]=le_encoder.fit_transform(final_data["AQI_Bucket"])
            import plotly.figure_factory as ff
            df_sample = final_data.iloc[0:10,0:10]
            colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
            font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
            fig =  ff.create_table(df_sample,colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])
            fig.show()
            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)
        
        def header(url):
            st.markdown(f'<p style="background-color:#87CEFA ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
                       Description of Data:<br>
                       The real-time data as collected from the field instruments is displayed live without human 
                       intervention from CPCB. 
                       It is likely that the live data may display some errors or abnormal values. 
                       Any abnormal value may be due to any episode or instrumental error at any particular time.
                       It contains Real time National Air Quality Index values from different monitoring stations across 
                       India. 
                       The pollutants monitored are Sulphur Dioxide (SO2), Nitrogen Dioxide (NO2), 
                       Particulate Matter (PM10 and PM2.5) , Carbon Monoxide (CO), Ozone(O3) etc
		  """
        
		
        header(html_temp11)

        st.markdown("""
                #### Basic  Tasks:
                + App covers the most basic Machine Learning task of Analysis, Correlation between variables.
                
                
                
                #### Machine Learning:
                + Machine Learning on different Machne Algorithms, modeling with different classifier and lastly  prediction. 
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            about_url="https://assets6.lottiefiles.com/packages/lf20_v1yudlrx.json"
            about_file12356=load_lottieurl(about_url)
            st_lottie(about_file12356 ,key ="about12658")
        
        #st.image('iidt_logo_137.png',use_column_width=True)
        st.markdown("<h2 style='text-align: center;'> This  is a Airquality Prediction Project :</h2>", unsafe_allow_html=True)

        st.markdown("""
                    #### @Author  Mr. Siddhartha Sarkar)
        
                    """)
        image2= Image.open("img1.jpg")
        st.image(image2,use_column_width=True)
        st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    
    #Horizontal Project selected

    if selected == "Projects":
            #st.title(f"You have selected {selected}")
            with st.sidebar:
                st.title("Project")
                project_url="https://assets6.lottiefiles.com/packages/lf20_rycdh53q.json"
                project_file12=load_lottieurl(project_url)
                st_lottie(project_file12 ,key="proj12")
            
            image2= Image.open("Understanding.jpg")
            st.image(image2,use_column_width=True)
            
            st.sidebar.title("Navigation")
            menu_list1 = ['Exploratoriy Data Analysis',"Prediction With Machine Learning"]
            menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if menu_Pre_Exp == 'Exploratoriy Data Analysis' and selected == "Projects":
                    st.title('Exploratoriy Data Analysis')

                    
                    
                    menu_list2 = ['None', 'Analysis']
                    menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)

                    
                    if menu_Exp == 'None':
                        st.markdown("""
                                    #### Kindly select from left Menu.
                                   # """)
                    
                    elif menu_Exp == 'Analysis':
                        label_analysis()

                    


            elif menu_Pre_Exp == "Prediction With Machine Learning" and selected == "Projects":
                    st.title('Prediction With Machine Learning')
                    
                    menu_list3 = ['Checking ML Method And Accuracy' ,'Checking Regression Method And Accuracy' ,'Prediction' ]
                    menu_Pre = st.radio("Menu Prediction", menu_list3)
                    
                    #Checking ML Method And Accuracy
                    if menu_Pre == 'Checking ML Method And Accuracy':
                            st.title('Checking Accuracy On Different Algorithms')
                            final_data=pd.read_csv("final_cleaned_data.csv")
                            final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
                            
                            if st.checkbox("View data"):
                                st.write(final_data)
                            model = st.selectbox("ML Method",['Logistic Regression', 'XGB Classifier', 'Random Forest Classifier'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression 
                                if model=='Logistic Regression':
                                    logistic_regression(get_data_class(final_data))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                elif model=='XGB Classifier':
                                    xgb_classifier(get_data_class(final_data))                                    
                                
                               
                                
                                
                                #Random Forest Classifier & CountVectorizer
                                elif model=='Random Forest Classifier':
                                    randomforest_classifier(get_data_class(final_data))
                              
                    #Checking ML Method And Accuracy
                    elif menu_Pre == 'Checking Regression Method And Accuracy':
                            st.title('Checking Accuracy On Different Algorithms')
                            final_data=pd.read_csv("final_cleaned_data.csv")
                            final_data=final_data.drop(columns=["Unnamed: 0"],axis=1)
                            
                            if st.checkbox("View data"):
                                st.write(final_data)
                            model = st.selectbox("ML Method",['XGboost_regressor', 'Random Forest Regressor'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression 
                                if model=='XGboost_regressor':
                                    XGboost_regressor(get_data_reg(final_data))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                elif model=='Random Forest Regressor':
                                    randomforest_regressor(get_data_reg(final_data))                                    
                                
                               
                                
                                
                                #Random Forest Classifier & CountVectorizer
                                #elif model=='Random Forest Classifier':
                                    #randomforest_classifier(get_data_class(final_data))
                                          
                    elif menu_Pre == 'Prediction':
                        st.title('Prediction')
                            
                        df= user_input_features()
                        
                        result_pred = predict_func(df)
                        
                        result_pred1=predict_func1(df)
                        st.success('The predicted LabelId is {}'.format(result_pred))
                        st.markdown("""
                                #### Air Quality Of Cities have Following Labelled Categories:
                                + 1. Moderate        [10868]
                                + 2. Satisfactory    [10074]
                                + 3. Poor             [2791] 
                                + 4. Very Poor        [2337]
                                + 5. Severe           [2013]
                                + 6. Good             [1448]
                                """)
                                
                        st.success('The Air Quality Index(AQI)  {}'.format(result_pred1))
                        if result_pred == 0:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)
                        elif result_pred == 1:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)    
                        elif result_pred == 2:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)    
                        elif result_pred == 3:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)    
                        elif result_pred == 4:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)
                        elif result_pred == 5:
                            image= Image.open("image.jpg")
                            st.image(image,use_column_width=True)
                            
                                
    if selected == "Report":
        #report_1
        st.title("Profile Report")
        st.sidebar.title("Project_Profile_Report")
        with st.sidebar:
            st_lottie(report_1, key="report1")
            #image= Image.open("report_project.png")
            #add_image=st.image(image,use_column_width=True)
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Simple EDA App with Streamlit Components</h1>
		</div>  """
        
		
        components.html(html_temp)
        st.sidebar.title("Navigation")
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            data_file= st.file_uploader("Upload CSV file",type=['csv'])
            if  data_file!= None:
                df = pd.read_csv(data_file)
                st.table(df.head(10))
                if st.button("Generate Profile Report"):
                    profile= ProfileReport(df)
                    st_profile_report(profile)
            
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            data_file = st.file_uploader("Upload CSV file",type=['csv'])
            if  data_file!= None:
                df = pd.read_csv(data_file)
                st.dataframe(df.head(10))
                if st.button("Generate Sweetviz Report"):
                    report = sv.analyze(df)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html")  
    			
		       
                
        
        
                                                      
if __name__=='__main__':
    main()            
            
            

