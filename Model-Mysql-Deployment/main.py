#copy mysql_connection.py into same folder as main.py
#Execute all lines of mysql_connection.py before executing this

#required libraries
from mysql_connection import *
from mysql.connector import errorcode
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
import time

#initializing credentials
config = {
  'user': 'root',
  'password': 'root',
  'host': '127.0.0.1',
  'raise_on_warnings': True
}

#checking for the connection
connecting(config)

############################################################################
#Name of DataBase and Table Name
DB_NAME = 'renewal_db'
table_name = 'renewal_table'

#defining function to create a database
#if db already exists try changing DB_NAME
def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'UTF8MB4'".format(DB_NAME))
        print("Database {} created successfully.".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        
#sleep for 3 sec
time.sleep(3)

#creating a dabase in MySQL server
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

##################################################################################

#loading dataset Final_Dataset.csv from google drive also can be loaded using local path
url = 'https://drive.google.com/file/d/1n__aCGFj89JcI3OFaN1Wyt7Hh0XD9iFk/view?usp=sharing'
url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
final_df = pd.read_csv(url2)

#sleep for 3 sec
time.sleep(3)
print("Data read from Google Drive")
####################################################################################
#writing data from csv to MySQL server
# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"  
                      .format(user="root", pw="root", 
                      db=DB_NAME))
# Insert whole DataFrame into MySQL
final_df.to_sql(table_name, con = engine, if_exists = 'replace', chunksize = 1000,index=False)

#sleep for 3 sec
time.sleep(3)
print("Data exported to MySQL Server")
# ######################################################################################
# #Query the database to check our work


# '''# # Fetch all the records
# # result = cursor.fetchall()
# # for i in result:
# #     print(i)
# #     print("\n")'''

# import tkinter  as tk 
# from tkinter import * 
# my_w = tk.Tk()
# my_w.geometry("1000x500")

# print("Displaying Data from MySQL Server")

# # Execute query
# sql = "SELECT * FROM {} limit 0,10".format(table_name)
# cursor.execute(sql)
# i=0
# j=0
# for table_name in cursor: 
#     for j in range(len(table_name)):
#         e = Entry(my_w, width=10, fg='blue') 
#         e.grid(row=i, column=j) 
#         e.insert(END, table_name[j])
#     i=i+1
# time.sleep(3)
# my_w.mainloop()
###############################################################################
#Retrieving data from MySQL

print("Retrieving Data from MySQL")
#sleep for 3 sec
time.sleep(3)

sql_select_Query = "SELECT * FROM {}".format(table_name)
cursor = cn.cursor()
cursor.execute(sql_select_Query)

# get all records
data_sql = pd.DataFrame(cursor.fetchall())
print("Data Retrieved from MySQL")
print("Total number of rows in table: ", cursor.rowcount)
#################################################################################
#scaling
from sklearn.preprocessing import scale
df_scale = scale(final_df)
########################Splitting the data #############
#splitting the data into Predictors and Target
X = final_df.iloc[:,:-1] # Predictors 
Y = final_df.iloc[:,-1] # Target
#sleep for 3 sec
time.sleep(3)
print("Performing EDA and Creating Model")
#######################KNN############################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, Y)
#sleep for 3 sec
time.sleep(3)
print("Fitting Model and Creating Pickel")
########################Pickle#########################
#loading model into pickel
import pickle
pickle.dump(knn,open('pckl_model.pkl','wb'))
#sleep for 3 sec
time.sleep(3)
print("Pickel file Created")
print("done..")
######################################END###############