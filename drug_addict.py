import simple_colors
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#) title
st.title(':orange[💮Student drug addict analysis]🌞')
#) reading the datset 1
df1 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\Students drugs Addiction Dataset\student_addiction_dataset_test.csv")
#df1.head(4)

#) dataframe 1 info
#df1.info()

#) dataframe 1 description
#df1.describe()

#) reading the dataset 2
df2 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\Students drugs Addiction Dataset\student_addiction_dataset_train.csv")
#df2.head(4)

#) dataframe 2 info
#df2.info()

#) dataframe 2 description
#df1.describe()

#) concatenation of 2 dataframes
df = pd.concat([df1, df2], axis=0, ignore_index=True)
st.subheader(":violet[1. student drug addict data Analysis:]\n")
if (st.button(':blue[click here]')):
        st.markdown("\n#### :red[1.1 student drug addict data:]")
        data = df.head(5)
        st.dataframe(data.style.applymap(lambda x: 'color:purple'))

#) dataframe info
#df.info()

#) check for the null values
#df.isna().sum()

#) filling the null values
df = df.ffill()
df = df.bfill()

#) check again for the null values
#df.isna().sum()

st.subheader(":violet[2. student drug addict data visualization:]\n")
if (st.checkbox("countplot-I")):
        #)countplot Academic_Performance_Decline column
        st.markdown("\n#### :red[2.1 countplot-I:]\n")
        fig = plt.figure(figsize=(15,8))
        df["Academic_Performance_Decline"].value_counts().plot(kind = 'bar')
        st.pyplot(fig)

if (st.checkbox("countplot-II")):
        #)countplot Social_Isolation column
        st.markdown("\n#### :red[2.2 countplot-II:]\n")
        fig = plt.figure(figsize=(15,8))
        df["Social_Isolation"].value_counts().plot(kind = 'bar')
        st.pyplot(fig)

if (st.checkbox("countplot-III")):
        #)Physical_Mental_Health_Problems
        fig = plt.figure(figsize=(15,8))
        df["Physical_Mental_Health_Problems"].value_counts().plot(kind = 'bar')
        st.pyplot(fig)

#)dummies
df = pd.get_dummies(df,columns=['Experimentation', 'Academic_Performance_Decline', 'Social_Isolation',
       'Financial_Issues', 'Physical_Mental_Health_Problems',
       'Legal_Consequences', 'Relationship_Strain', 'Risk_Taking_Behavior',
       'Withdrawal_Symptoms', 'Denial_and_Resistance_to_Treatment',
       'Addiction_Class'],drop_first=True)

#) Ml models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

X =df.drop(['Addiction_Class_Yes'],axis=1)
y = df['Addiction_Class_Yes']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()

model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
model_name = type(model).__name__

st.subheader(':violet[3.ML model]')
selectBox=st.selectbox("model: ", ['logistic',
                                       'feature',
                                       ])
if selectBox == 'logistic':
    st.markdown("\n#### :red[3.1 logistc regression:]")
    st.success(model_name)
    #)train
    st.write('**Train**')
    st.write(f":violet[Accuracy score: {accuracy_score(y_train,train_pred)}]")
    st.write(f":green[Precision: {precision_score(y_train,train_pred)}]")
    st.write(f":violet[Recall: {recall_score(y_train,train_pred)}]")
    st.write(f":green[F1 Score: {f1_score(y_train,train_pred)}]")
    #)test
    st.write('**Test**')
    st.write(f":violet[Accuracy score: {accuracy_score(y_train,train_pred)}]")
    st.write(f":green[Precision: {precision_score(y_train,train_pred)}]")
    st.write(f":violet[Recall: {recall_score(y_train,train_pred)}]")
    st.write(f":green[F1 Score: {f1_score(y_train,train_pred)}]")

#)feature importance
import numpy as np
from sklearn.linear_model import LogisticRegression

#) initialize logistic regression model and fit to training data
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

#) obtain coefficients
coefficients = logreg.coef_.reshape(-1)

#) present features and absolute values of coefficients in a training dataframe
logreg_fit_df = pd.DataFrame(data = {'Feature':X.columns,'Importance':np.abs(coefficients)}).sort_values(by ='Importance',ascending = False)


if selectBox == 'feature':
    st.markdown("\n#### :red[3.2 feature importance dataframe:]")
    data = logreg_fit_df
    st.dataframe(data.style.applymap(lambda x: 'color:green'))