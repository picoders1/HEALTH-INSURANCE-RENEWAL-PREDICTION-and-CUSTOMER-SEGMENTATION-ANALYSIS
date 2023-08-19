
'''install connector in anaconda prompt (environment:base) using below command'''

#pip install mysql-connector-python

import mysql.connector
from mysql.connector import errorcode

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
