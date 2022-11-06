############################# MySQL Deployment #############################

'''Packages Required for SQL'''

import mysql.connector                 #API for MySQL/Python
from mysql.connector import errorcode
from sqlalchemy import create_engine   #For Table creation in MySQL
import time
import pandas as pd


#Function for Connection
def connecting(config):
    try:
      cnx = mysql.connector.connect(**config)
      print("Connected to MySQL server")
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
      else:
        print(err)
    else:
      cnx.close()

# MySQL Server Details
config = {'user': 'root','password': 'root','host': '127.0.0.1','raise_on_warnings': True}


#Establishing Connection
print("Establishing Connection with MySQL Server...")
time.sleep(2)
connecting(config)

############################################################################

#Name of DataBase and Table Name
DB_NAME = 'medical_insurance_db'
table_name = 'medical_insurance_table'

#Function for Database & Table 
def create_database(cursor):
    try:
        cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'UTF8MB4'".format(DB_NAME))
        print("Database {} created successfully.".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        
        
#Database and Table Creation

try:
    cn = mysql.connector.connect(**config)
    cursor = cn.cursor(buffered=True)
    cursor.execute("USE {}".format(DB_NAME))
    print("Database {} already exists, Using {}!".format(DB_NAME,DB_NAME))
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        cn.database = DB_NAME
    else:
        print(err)

time.sleep(2)        
print("Created {} Database and empty {} ...".format(DB_NAME,table_name))

############################################################################

#Data Downloading from Google Drive
time.sleep(2)
print("Reading Data from Google Drive")

url = 'https://drive.google.com/file/d/1g7m0LLsrOn-vu0hv_ecBPBIOzOdLKj7f/view?usp=sharing'
url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url2)

####################################################################################

#Loading data in MySQL
time.sleep(2)
print("Loading Data in the empty {} in {} database".format(DB_NAME,table_name))

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw="root",db=DB_NAME))

# Insert whole DataFrame into MySQL
df.to_sql(table_name, con = engine, if_exists = 'replace', chunksize = 1000,index=False)

time.sleep(2)
print("Data exported to MySQL Server")

###############################################################################

#Retrieving data from MySQL
print("Retrieving {} from MySQL".format(table_name))
time.sleep(2)

sql_select_Query = "SELECT * FROM {}".format(table_name)
cursor = cn.cursor()
cursor.execute(sql_select_Query)

# Fetching all Records in the table
data_sql = pd.DataFrame(cursor.fetchall())
print("Data Retrieved from MySQL")
print("Total number of rows in table: ", cursor.rowcount)

######################## Modelling ########################

time.sleep(2)
print("Fitting Model and Creating Pickel")

#Independent and Dependent Variables
X = data_sql.iloc[:,:-1] # Predictors 
Y = data_sql.iloc[:,-1] # Target

from sklearn.linear_model import LogisticRegression

logit=LogisticRegression().fit(X,Y)

######################## Pickling ########################

import pickle
pickle.dump(logit,open('predict_model.pkl','wb'))

time.sleep(2)
print("Pickel file Created")
time.sleep(2)
print("File Executed Successfully")






