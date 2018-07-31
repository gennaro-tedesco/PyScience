import mysql.connector
import pandas as pd
import mysql_connector.db_config as cfg
from mysql.connector import errorcode


def get_connector():
	try:
		return mysql.connector.connect(**cfg.config)
	except mysql.connector.Error as err:
		if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
			print("Wrong user name or password")
		elif err.errno == errorcode.ER_BAD_DB_ERROR:
			print("Database does not exist")
		else:
			print(err)


def get_query(file_name):
	fd = open(file_name, 'r')
	sql_file = fd.read()
	fd.close()
	return sql_file


def run_query(query, cnx):
	assert (isinstance(query, str))
	assert (cnx is not None)
	cur = cnx.cursor()
	cur.execute(query)
	columns = [i[0] if type(i[0]) is str else i[0].decode('utf-8') for i in cur.description]
	return pd.DataFrame(cur.fetchall(), columns=columns)
