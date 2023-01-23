import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt # import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import shap
from sklearn.cluster import KMeans
from zipfile import ZipFile
plt.style.use('fivethirtyeight')


@st.cache
def load_data():
	
    z = ZipFile('train_sample_30mskoriginal.zip') 
    data = pd.read_csv(z.open('train_sample_30mskoriginal.csv'), index_col='SK_ID_CURR', encoding ='utf-8') # data = pd.read_csv(z.open('X_data_rfecv_32.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    data = data.drop('Unnamed: 0', axis=1)
    z = ZipFile('train_sample_30m.zip')
    sample = pd.read_csv(z.open('train_sample_30m.csv'), index_col='SK_ID_CURR', encoding ='utf-8') 
    
    description = pd.read_csv('HomeCredit_columns_description.csv', 
    				usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
    				
    target = data.TARGET # target = data.iloc[:, -1:]
    
    return data, sample, target, description

# data = load_data() # --> for displaying purpose --> Not to be done if data is heavy, OK if size data < 200 MB
# st.write(data)


def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier_best_customscore.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_gen_info(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]


    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets
    
    
def client_identity(data, id):
    data_client = data[data.index == int(id)]
    return data_client    


@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age


@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income


@st.cache
def load_prediction(sample, id, clf):
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score


@st.cache
def load_kmeans(sample, id, mdl):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)


@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2).fit(sample)
    return knn 


# Loading data
data, sample, target, description = load_data()
id_client = sample.index.values
clf = load_model()



#******************************************
# MAIN
#******************************************

# Title display
html_temp = """
<div style="background-color: LightSeaGreen; padding:5px; border-radius:10px">
	<h1 style="color: white; text-align:center">Credit Allocation Dashboard</h1>
</div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

# Customer ID selection
# st.sidebar.header('General Informations')

# Loading selectbox
chk_id = st.sidebar.selectbox('Client ID', id_client)

# Loading general informations
nb_credits, rev_moy, credits_moy, targets = load_gen_info(data)



#*******************************************
# Displaying informations on the sidebar
#*******************************************

# Number of loans for clients in study
st.sidebar.markdown("<u>Total number of loans in our sample :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average income
st.sidebar.markdown("<u>Average income ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>Average loan amount ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)


# PieChart
fig, ax = plt.subplots(figsize=(5,5))
plt.pie(targets, explode=[0, 0.5], labels=['Reimbursed', 'Defaulted'], autopct='%1.1f%%', startangle=45)
st.sidebar.pyplot(fig)


#******************************************
# MAIN -- suite
#******************************************

# Display Customer ID from Sidebar
st.write('Customer ID selected :', chk_id)

# Displaying customer information : gender, age, family status, Nb of hildren etc.
st.subheader('Customer general informations')

if st.checkbox("Enable (Disable) customer summary"):
   infos_client = client_identity(data, chk_id)
   st.write("Gender : ", infos_client["CODE_GENDER"].values[0])
   st.write("Age : {:.0f} years old".format(int(infos_client["DAYS_BIRTH"]/365)))
   st.write("Family status : ", infos_client["NAME_FAMILY_STATUS"].values[0])
   st.write("Number of children : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
   
   # Age distribution plot
   data_age = load_age_population(data)
   fig, ax = plt.subplots(figsize=(10, 5))
   sn.histplot(data_age, edgecolor = 'b', color="orange", bins=15)
   ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
   ax.set(title='Customer age', xlabel='Age (Year)', ylabel='')
   st.pyplot(fig)
   
   st.subheader("Customer pecuniary informations ($US)")
   st.write("Income total : {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
   st.write("Credit amount : {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
   st.write("Credit annuities : {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
   st.write("Amount of property for credit : {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

   # Income distribution plot
   data_income = load_income_population(data)
   fig, ax = plt.subplots(figsize=(10, 5))
   sn.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'g', color="orange", bins=10)
   ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
   ax.set(title='Customer income', xlabel='Income ($US)', ylabel='')
   st.pyplot(fig)
	
else:
  st.markdown("<i>…</i>", unsafe_allow_html=True)


# Customer solvability display
st.header("Customer report")
prediction = load_prediction(sample, chk_id, clf)
st.write("Default probability : {:.0f} %".format(round(float(prediction)*100, 2)))

st.markdown("<u>Customer data :</u>", unsafe_allow_html=True)
st.write(client_identity(data, chk_id))    
    

#Feature importance / description
if st.checkbox("Show (Hide) customer #{:.0f} feature importance".format(chk_id)):
   shap.initjs()
   X = sample.iloc[:, :-1] # X = sample.loc[:, sample.columns != 'TARGET']
   X = X[X.index == chk_id]
   number = st.slider("Pick a number of features", 0, 20, 5)

   fig, ax = plt.subplots(figsize=(9,9))
   explainer = shap.TreeExplainer(load_model())
   shap_values = explainer.shap_values(X)
   shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
   st.pyplot(fig)
        
   if st.checkbox("Select feature for desciption") :
      list_features = description.index.to_list()
      feature = st.selectbox('Feature checklist', list_features)
      st.table(description.loc[description.index == feature][:1])
   
else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)
    
    
# Similar customer to the one selected
neighbors_nearest = st.checkbox("Show (Hide) similar customer")

if neighbors_nearest:
   knn = load_knn(sample)
   st.markdown("<u>10 closest customers to the selected one :</u>", unsafe_allow_html=True)
   st.dataframe(load_kmeans(sample, chk_id, knn))
   st.markdown("<i>Target 1 = Customer high default probability</i>", unsafe_allow_html=True)
   
else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)    
