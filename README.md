# HEALTH INSURANCE RENEWAL PREDICTION and CUSTOMER SEGMENTATION ANALYSIS

## 1. Medical Insurance

To be financially stable is everyone’s dream. However, the rise in medical costs and treatment for illness can strain your savings. Health insurance helps lessen the costs of medical expenses in the event of an illness or accident and for preventive medicine such as routine medical tests, check-ups, and screening tests. The main benefits are cashless treatment, Pre and post-hospitalization cost coverage, Transportation facility, No Claim Bonus (NCB), Medical checkups, Room rent, and tax benefits. Health insurance is typically offered as one- to three-year contracts and requires renewal based on the chosen plan.

## Objectives:
1. The objective is to predict insurance policy renewals for existing customers using predictive modeling.
2. Segment existing customers for better targeting through focused marketing strategies.


## 2. Technical Stacks

These are the Software, Tools, and Environments used in the project.

HTML, CSS, JS: Cascading Style Sheets (CSS) are used for presenting documents written in a markup language such as HTML. CSS is a cornerstone technology of the World Wide Web, alongside HTML and JavaScript.

Flask: Flask helps end users interact with your Python code (in this case, our ML models) directly from their web browser without needing any libraries or code files.

Tableau: A tool we used for visualization.

MySQL: A tool used for storing databases.

Heroku: A tool used for Deploying the model.


## 3. Project Architecture / Data Pipeline

![image](https://github.com/picoders1/HEALTH-INSURANCE-RENEWAL-PREDICTION-and-CUSTOMER-SEGMENTATION-ANALYSIS/assets/87698874/03fdd05e-d39a-4c71-be79-9f112aa4f452)


## 4. Data Understanding

The first step in data understanding is Data Collection. The feature set used in this project was taken from the MySQL database, containing 42 features related to the health insurance renewal policy. The output variable for this project is “Renewal: Yes or No”, which is a discrete data type. So this project will focus on the classification machine learning algorithm.


## 5. Exploratory Data Analysis(EDA)

1. Data Cleansing is a primary process that needs to be worked on after data collection.
2. We have performed outlier treatment on features that had outliers, as outliers were affecting the mean values.
3. Dummy variables were also created for categorical variables using Label Encoding.
4. We have also performed Standardization of data.
5. Business Moments Decisions and graphical interpretation of data are performed before and after data cleansing to analyze the statistics of the data
6. Visualization of univariate and bivariate plots was done in Python and Tableau.


## 6. Model Building 

### Classification Algorithms: 
1. Shallow Model(KNN, Naive Bayes, Decision Tree) 
2. Ensemble Model(Random Forest) 
3. Regression Model(Logistic Regression) 

### Algorithms Used
1. Support Vector Machine(SVM) 
2. Artificial Neural Network(ANN) 

### Model Segmentation
1. Hierarchical Clustering 
2. Density-Based Clustering of Application with Noise 
3. K-Means Clustering


## 7. Model Evaluation

Model Hyper-parameters used:
1. Cross Validation 
2. GridSearchCV 
3. RandomSearchCV 

Model Accuracy Measures :
1. Confusion matrix 
2. Accuracy 
3. F1 score 
4. ROC (Receiver Operating Characteristics) curve & AUC (Area Under Curve) 


## 8. Deployment Strategy

1. Flask : 
1. Flask is a micro-framework for building web applications in Python. It began as a simple wrapper around Wekzeug(WSGI protocol) and Jinja and has become    
   the most popular Python web application Framework.
2. The Flask and Green-unicorn module must be installed in the project environment using 'pip install flask gunicorn.
3. Gunicorn is a Python WSGI HTTPS Server that uses a worker model.

2. Heroku :
2.1 Heroku is a cloud platform as a service (PaaS) supporting several programming languages.


## 9. Conclusion

1. This project is an exploratory attempt to understand the factors that affect renewal decisions in the health insurance market.
2. The prediction model helped us to figure out features that contributed more to the renewal of health insurance policies.
3. Customer segmentation provided further insights into the business, which would provide a targeted marketing  approach.
4. The results also suggest customer satisfaction is a significant factor in influencing the renewal decision of policyholders.


## Thank You !!
