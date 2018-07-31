import os

config = {
	'user': os.environ['DB_USER'],
	'password': os.environ['DB_PASSWORD'],
	'database': os.environ['DB_DATABASE'],
	'host': os.environ['DB_HOST']
}
