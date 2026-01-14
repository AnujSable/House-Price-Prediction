#### **🏠 House Price Prediction System – End-to-End Machine Learning Project**



**A production-grade Machine Learning system that predicts house prices and deploys the trained model using a real-time Streamlit web application.**



##### **📌 Project Overview:-**



This project implements a full ML lifecycle pipeline – from raw dataset to a deployed web application.



The system predicts the price of a house based on multiple real-world features such as:



Area (sq ft)



Bedrooms, Bathrooms, Floors



Year Built → converted to House Age



Location, Condition, Garage availability



This is not just a model – it is a deployable ML product following industry practices.





##### **🎯 Objectives:-**



* Build a regression model to accurately predict house prices.
* Perform feature engineering and preprocessing.
* Train and tune a Random Forest model using GridSearchCV.
* Persist the model along with scaler and schema.
* Deploy the system using Streamlit with zero feature-mismatch risk.





##### **🧠 Technologies Used:-**



     **•Category	           •Tools**

Programming Language	Python

Data Processing	Pandas

Machine Learning	Scikit-Learn

Algorithm		Random Forest Regressor

Model Tuning		GridSearchCV

Feature Scaling		StandardScaler

Model Storage		Joblib

Deployment		Streamlit





##### **📂 Project Structure:-**



house-price-prediction/

│

├── Realistic\_House\_Price\_Dataset.csv

├── train\_model.py

├── app.py

├── house\_price\_model.pkl

├── scaler.pkl

├── model\_columns.pkl

├── requirements.txt

└── README.md





##### **🔄 Machine Learning Workflow:-**



**1️⃣ Data Preprocessing**



Column names normalized (lowercase, spaces removed).



Dropped unnecessary columns like id.



Converted yearbuilt into houseage.



**2️⃣ Feature Engineering**



Encoded categorical features using One-Hot Encoding:



Location



Condition



Garage



**3️⃣ Feature Scaling**



Numerical features scaled using StandardScaler:



\['area','bedrooms','bathrooms','floors','houseage','areaperbedroom']



**4️⃣ Model Training**



Used RandomForestRegressor.



Tuned hyperparameters using GridSearchCV.



Best model selected automatically based on R² score.



**5️⃣ Model Persistence (Critical Design)**



The following are saved after training:



joblib.dump(best\_model, "house\_price\_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(list(X.columns), "model\_columns.pkl")





This ensures training schema = prediction schema permanently.





##### **🌐 Web Deployment with Streamlit:-**



The web app dynamically:

* Loads the trained model
* Loads the saved scaler
* Loads exact feature columns
* Builds input dynamically based on training schema





##### **🚀 How to Run the Project:-**



**Install dependencies:-**

pip install -r requirements.txt



**Train the model:-**

python train\_model.py

&nbsp;

**Launch the application:-**

venv\\Scripts\\activate

streamlit run app.py





###### **📈 Sample Output:-**

**Estimated House Price: ₹ 6,450,000**





##### **🏆 Why This Project Is Special:-**



✔ End-to-end ML product

✔ Automatic hyperparameter tuning

✔ Zero feature mismatch design

✔ Production-safe schema binding

✔ Real-time interactive UI

