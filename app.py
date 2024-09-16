import streamlit as st
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import joblib
import shap
import matplotlib.pyplot as plt

def introduce_page():

    st.write("# Welcome to use！")

    st.sidebar.success("Click 👈 predict the droplet size")

    st.markdown(
        """
        # Prediction of droplet size in capillary embedded microchannels
       This application utilizes machine learning models to predict the size of microdispersed droplets or bubbles.

        ## Introduction
        - Objective: Help lab experimenters or engineers prepare user-specified droplets or bubbles.
        - Model algorithms: Random forest, Gradient boosting decision tree, XGBoost and Stacking

        ## Guidance for use
        - Inputting accurate and complete experimental information makes size predictions more accurate .
        - The predicted results can serve as an important reference for microdispersion experiments, but prudent decision-making is required.
        - Feel free to contact our technical support with any questions.

    """
    )


def predict_page():

    st.markdown(
        """
        ## Guidance for use
        This application utilizes machine learning models to predict the size of microdispersed droplets or bubbles.
        - 👉 **Input**：Input detailed and correct information of the microchannel, continuous phase, dispersed phase, and operating condition below.
        - 👉 **Prediction**：Predict the size of droplets or bubbles.
    """
    )

    with st.form('user_inputs'):
        
        method = st.radio("Cutting method", ("cross flow", "collisional flow"))
        device = st.radio("Type of the microchannel", ("Bending microchannel", "Reverse bending microchannel", "Straight microchannel", "Stepwise microchannel"))
        curvature = st.number_input('Curvature of the continuous microchannel', min_value=0.00)
        width = st.number_input('Width of the continuous microchannel (mm)', min_value=0.00)
        height = st.number_input('Height of the continuous microchannel (mm)', min_value=0.00)
        length = st.number_input('Distance of the capillary tip to corresponding microchannel wall (mm)', min_value=0.00)
        A3 = st.number_input('Internal cross-sectional area of the capillary (mm$^2$)', min_value=0.00)
        rhod = st.number_input('Density of the dispersed phase (kg/m$^3$)', min_value=0.00)
        rhoc = st.number_input('Density of the continuous phase (kg/m$^3$)', min_value=0.00)
        mud = st.number_input('Viscosity of the dispersed phase (mPa·s)', min_value=0.00)
        muc = st.number_input('Viscosity of the continuous phase (mPa·s)', min_value=0.00)
        gamma = st.number_input('Interfacial tension of the continuous phase (mN/m)', min_value=0.00)
        Qd = st.number_input('Dispersed phase flow rate (μL/min)', min_value=0.00)
        Qc = st.number_input('Continuous phase flow rate (μL/min)', min_value=0.00)


        submitted = st.form_submit_button('Prediction')
    if submitted:
        A1 = width * height
        WH = width / height
        A2A1= (length * height) /  (width * height)
        ratio_mu = mud / muc
        ud = (Qd * 0.001 / 60) / A3
        uc = (Qc * 0.001 / 60) / (length * height)
        ratio_Q = Qd / Qc
        ratio_u = ud / uc
        cutting = uc * muc
        format_data = [method, device, curvature, width, height, length, A3, A1, WH, A2A1, rhod, rhoc,
                       mud, muc, ratio_mu, gamma, Qd, Qc, ud, uc, ratio_Q, ratio_u, cutting]


        method = 0

        if method == 'cross flow':
            method = 1
        elif method == 'collisional flow':
            method = 0

        device1, device2, device3, device4 = 0, 0, 0, 0

        if device == 'Bending microchannel':
            device1 = 1
        elif device == 'Reverse bending microchannel':
            device2 = 1
        elif device == 'Straight microchannel':
            device3 = 1
        elif device == 'Stepwise microchannel':
            device4 = 1

        format_data = [method, device1, device2, device3, device4, curvature, width, height, length, A3, A1, WH, A2A1, rhod, rhoc,
                       mud, muc, ratio_mu, gamma, Qd, Qc, ud, uc, ratio_Q, ratio_u, cutting]


    rf_reg = joblib.load("rf.dat")
    gbdt_reg = joblib.load("gbdt.dat")
    xgb_reg = joblib.load("xgb.dat")

    df2_backup = pd.read_csv("data_model.csv", index_col='Unnamed: 0')
    df2 = deepcopy(df2_backup)
    X, y = df2.iloc[:, :-1], df2.iloc[:, -1]

    X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=1)
    

    def train_cross(X_train, 
                y_train, 
                X_test, 
                estimators, 
                n_splits = 5, 
                random_state = 1,
                blending = False,
                regress = False):

        if type(y_train) == np.ndarray:
            y_train = pd.Series(y_train)
    
        if blending == True:
            X, X1, y, y1 = tts(X_train, y_train, test_size=test_size, random_state=random_state)
            m = X1.shape[0]
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            X1 = X1.reset_index(drop=True)
            y1 = y1.reset_index(drop=True)
        else:
            m = X_train.shape[0]
            X = X_train.reset_index(drop=True) 
            y = y_train.reset_index(drop=True)
    
        n = len(estimators)
        m_test = X_test.shape[0] 
    
        columns = []
        for estimator in estimators:
            columns.append(estimator[0] + '_oof') 
    
        train_oof = pd.DataFrame(np.zeros((m, n)), columns=columns) 
    
        columns = []
        for estimator in estimators:
            columns.append(estimator[0] + '_predict')
    
        test_predict = pd.DataFrame(np.zeros((m_test, n)), columns=columns)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for estimator in estimators:
            model = estimator[1] 
            oof_colName = estimator[0] + '_oof'
            predict_colName = estimator[0] + '_predict'
        
            for train_part_index, eval_index in kf.split(X, y): 
                X_train_part = X.loc[train_part_index]
                y_train_part = y.loc[train_part_index]
                model.fit(X_train_part, y_train_part) 

                if regress == True:
                    if blending == True:
                        train_oof[oof_colName] += model.predict(X1) / n_splits
                        test_predict[predict_colName] += model.predict(X_test) / n_splits
                    else:
                        X_eval_part = X.loc[eval_index]
                        train_oof[oof_colName].loc[eval_index] = model.predict(X_eval_part) 
                        test_predict[predict_colName] += model.predict(X_test) / n_splits
            
                else:
                    if blending == True:
                        train_oof[oof_colName] += model.predict_proba(X1)[:, 1] / n_splits
                        test_predict[predict_colName] += model.predict_proba(X_test)[:, 1] / n_splits
                    else:
                        X_eval_part = X.loc[eval_index]
                        train_oof[oof_colName].loc[eval_index] = model.predict_proba(X_eval_part)[:, 1]
                        test_predict[predict_colName] += model.predict_proba(X_test)[:, 1] / n_splits
    
        if blending == True:
            train_oof[y1.name] = y1
        else:
            train_oof[y.name] = y
        
        return train_oof, test_predict
    
    if submitted:
        format_data_df = pd.DataFrame(data=[format_data], columns=rf_reg.feature_names_in_)


        predict_result_rf = rf_reg.predict(format_data_df)[0] * (2 * width * height / (width + height))
        predict_result_gbdt = gbdt_reg.predict(format_data_df)[0] * (2 * width * height / (width + height))
        predict_result_xg = xgb_reg.predict(format_data_df)[0] * (2 * width * height / (width + height))
        
        estimators = [("rf", rf_reg), ("gbdt", gbdt_reg), ("xgb", xgb_reg)] 
        train_oof, test_predict = train_cross(X_train, y_train, format_data_df, estimators, regress=True)
        lr_reg = LinearRegression().fit(np.array(train_oof.iloc[:, :3]), y_train)
        predict_result_st = lr_reg.predict(np.array(test_predict))[0] * (2 * width * height / (width + height))

        st.subheader('Based on the data you have entered, the predictions of the size of microfluidic droplet or bubble are:')
        st.write('Prediction of Random forest model: ', round(predict_result_rf, 2), 'mm')
        st.write('Prediction of Gradient boosting decision tree model: ', round(predict_result_gbdt, 2), 'mm')
        st.write('Prediction of XGBoost model: ', round(predict_result_xg, 2), 'mm')
        st.write('Prediction of Stacking model: ', round(predict_result_st, 2), 'mm')
        
        st.write('')
        st.subheader("Visualizing the Interpretation of XGBoost Model Prediction ($d_\mathrm{av}$/$d_\mathrm{e}$):")
        st.write(' Note: $d_\mathrm{e}=(W+H)/2(WH)$')
        shap.initjs()
        ex_xgb = shap.TreeExplainer(xgb_reg)
        shap_values_xgb = ex_xgb(format_data_df
                            ,check_additivity=False
                            )
        
        fig = shap.plots.waterfall(shap_values_xgb[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig, dpi=600)





st.set_page_config(
    page_title="Microfluidic droplet/bubble size prediction",
    page_icon="🔍️",
)

nav = st.sidebar.radio("Navigator", ["Introduction", "Prediction"])

if nav == "Introduction":
    introduce_page()
else:
    predict_page()
