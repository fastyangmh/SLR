from pymongo import MongoClient
from conf import database
def get_db(host=database.host,
        user=database.user,
        pwd=database.pwd,
        dbname=database.dbname,rs=database.rs):
    '''get db'''
    db = MongoClient(host=host, replicaset=rs)[dbname]
    if user != '':
        db.authenticate(user, pwd)
    return db
