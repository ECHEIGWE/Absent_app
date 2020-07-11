# Core Pkgs
import streamlit as st 


# EDA Pkgs
import pandas as pd 
import numpy as np 


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

#DB
from managed_db import *
# passoword
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password, hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

best_features = ['work_load_average/day_','transportation_expense','reason_for_absence', 'disciplinary_failure', 'son', 'pet','distance_from_residence_to_work', 'day_of_the_week','service_time', 'social_drinker', 'social_smoker', 'age']
d_reason_for_absence = {"Certain infectious and parasitic diseases":1, 
"Neoplasms":2,
"Diseases of the blood and blood-forming organs":3, 
"Endocrine, nutritional and metabolic diseases":4,"Mental and behavioural disorders":5,
"Diseases of the nervous system":6,
"Diseases of the eye and adnexa":7,
"Diseases of the ear and mastoid process":8,
"Diseases of the circulatory system":9,
"Diseases of the respiratory system":10,
"Diseases of the digestive system":11,
"Diseases of the skin and subcutaneous tissue":12,
"Diseases of the musculoskeletal system and connective tissue":13,
"Diseases of the genitourinary system":14,
"Pregnancy, childbirth and the puerperium":15,
"Certain conditions originating in the perinatal period":16,
"Congenital malformations, deformations and chromosomal abnormalities":17,
"Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified":18,
"Injury, poisoning and certain other consequences of external causes":19,
"External causes of morbidity and mortality":20,
"Factors influencing health status and contact with health services":21,

"patient follow-up":22, "medical consultation":23, "blood donation":24, "laboratory examination":25, 
"unjustified absence":26, "physiotherapy":27, "dental consultation":28}

d_disciplinary_failure = {"Yes":1, "No":0}
d_day_of_the_week = {"Monday":2, "Tuesday":3, "Wednesday":4, "Thursday":5, "Friday":6}
d_social_drinker = {"Yes":1, "No":0}
d_social_smoker = {"Yes":1, "No":1}

def get_value(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

def get_key(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return key
  

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


 

def main():
	"""Absenteeism Prediction App"""
	st.title("Absenteeism Prediction App")

	menu = ["Home", "Login", "Signup"]
	submenu = ["Plot", "Prediction"]

	choice = st.sidebar.selectbox("Menu", menu)
	if choice == "Home":
		st.subheader("Home")
		st.markdown(""" 
			**What is Absenteeism**: Absence from work during normal working hours resulting in temporary 
			incapacity to execute a regular working activity.

			* **Based on information provided we can predict whether an employee is expected to be absent or not?**

			* ** Data Source:** UCI Database
			""")
	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password", type = 'password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))
			#if password == "12345":
			if result:
				st.success("Welcome {}".format(username))
			activity = st.selectbox("Activity", submenu)
			if activity == "Plot":
				st.subheader("Data Vis Plot")
				df = pd.read_csv("data/new_absent_data.csv")
				st.dataframe(df)
				#value count of target
				st.text('VALUE COUNT OF THE TARGET')
				df['absenteeism_time_in_hours'].value_counts().plot(kind='bar')
				st.pyplot()
				#Freq dis plot
				st.text('FREQUECY DISTRIBUTION OF AGE')
				freq_df = pd.read_csv('data/feq_absent_data.csv')
				st.bar_chart(freq_df['count'])

				if st.checkbox('Area chart'):
					clean_colums = df.columns.to_list()
					feat_colums = st.multiselect("Choose a Feature", clean_colums)
					new_df = df[feat_colums]
					st.area_chart(new_df)



			elif activity =="Prediction":
				st.subheader("Predictive Analytics")
				work_load_average = st.number_input('Work_load', 205000,380000)
				transportation_expense = st.number_input('Transportation_expenses', 117,390)
				reason_for_absence = st.selectbox("Reason_for_Absence", tuple(d_reason_for_absence.keys()))
				disciplinary_failure = st.radio("Disciplinary_Failure", tuple(d_disciplinary_failure.keys()))
				son = st.number_input("Number of Children", 0.,4.)
				pet = st.number_input("Pet", 0., 8.)
				distance_from_residence_to_work = st.number_input("Distance_from_residence_to_work(KM)", 5,52)
				day_of_the_week = st.selectbox("Day_of_the_week", tuple(d_day_of_the_week.keys()))
				service_time = st.number_input("Service Time", 1, 29)
				social_drinker = st.radio("A Social_Drinker?", tuple(d_social_drinker.keys()))
				social_smoker = st.radio("A Social_Smoker?", tuple(d_social_smoker.keys()))
				age = st.number_input("Age", 27, 58)
				#USER INPUT 


				k_reason = get_value(reason_for_absence,d_reason_for_absence)
				k_disciplinary = get_value (disciplinary_failure, d_disciplinary_failure)
				k_day_of_the_week = get_value(day_of_the_week,d_day_of_the_week)
				k_social_drinker = get_value(social_drinker, d_social_drinker)
				k_smoker = get_value(social_smoker, d_social_smoker)
				#USER RESULT


				selected = [work_load_average,transportation_expense,reason_for_absence,disciplinary_failure,son,pet,distance_from_residence_to_work,day_of_the_week,service_time,social_drinker,social_smoker,age]
				vectorized = [work_load_average,transportation_expense,k_reason,k_disciplinary, son,pet,distance_from_residence_to_work,k_day_of_the_week,service_time, k_social_drinker, k_smoker,age]
				sample_data = np.array(vectorized).reshape(1,-1)
				st.info(selected)
				json_form = {"Work_load": work_load_average, "Transportation_expenses": transportation_expense, "Reason_for_Absence": reason_for_absence,
				"Disciplinary_Failure": disciplinary_failure, "No of Children": son, "Pet": pet, "Distance_from_residence_to_work(KM)": distance_from_residence_to_work,
				"Day_of_the_week": day_of_the_week, "Service_Time": service_time, "Social Drinker": social_drinker, "Social_Smoker":social_smoker,"Age":age}
				st.json(json_form)
				#st.write(vectorized)

				#MODEL EVALUTION
				my_model_list = ("Logistic ReGression","KNeighours","Ramdon Forest")
				model_choice = st.selectbox("Model Choice", my_model_list)
				if st.button('Predict'):
										
					if model_choice =='Logistic ReGression':
						predictor = load_model("models/logistic_Absent_model.pkl")
						prediction = predictor.predict(sample_data)
						pred_pro = predictor.predict_proba(sample_data)
					#if model_choice =='SVM Classifier':
						#predictor = load_model("models/clf_Absent_model.pkl")
						#prediction = predictor.predict(sample_data)
						#pred_pro = predictor.predict_proba(sample_data)
					if model_choice =='KNeighours':
						predictor = load_model("models/knn_Absent_model.pkl")
						prediction = predictor.predict(sample_data)
						pred_pro = predictor.predict_proba(sample_data)
					if model_choice =='Ramdon Forest':
						predictor = load_model("models/rfc_Absent_model.pkl")
						prediction = predictor.predict(sample_data)
						pred_pro = predictor.predict_proba(sample_data)


					if prediction == 1:
						st.warning("Individal will be Absent")
						pred_probability_score = {"Absent":pred_pro[0][0]*100,"Present":pred_pro[0][1]*100}
						st.subheader("Prediction Probability Score using {}".format(model_choice))
						st.json(pred_probability_score)
					else:
						st.success("Individal will be Present")
						pred_probability_score = {"Absent":pred_pro[0][0]*100,"Present":pred_pro[0][1]*100}
						st.subheader("Prediction Probability Score using {}".format(model_choice))
						st.json(pred_probability_score)



	 

	elif choice == "Signup":
		new_username = st.text_input("User name")
		new_password = st.text_input("Password", type = 'password')

		confirm_password = st.text_input("Confirm Password", type = 'password')
		if new_password == confirm_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")
		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username, hashed_new_password)
			st.success ("You have successfully created a new account")
			st.info("Login to Get Started")





	 
        
                
                 
              
    

if __name__ == '__main__':
	main()