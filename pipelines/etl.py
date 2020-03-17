import petl as etl
import psycopg2 as pg
import sys
from sqlalchemy import *

dbConnections  = {
    "operations" : "dbname=operations user=etl host=1227.0.0.1",
    "python" : "dbname=python user=etl host=127.0.0.1"
}
        
srcConnections = pg.connect(dbConnections["operations"])
tgtConnection = pg.connect(dbConnections["python"])

srcCursor = srcConnections.cursor()
tgtCursor = tgtConnection.cursor()

query = """
SELECT table_name 
FROM information_schema.columns
WHERE table_schema = 'public'
GROUP BY 1
"""
srcCursor.execute(query)
srcTables = srcCursor.fetchall()

for table in srcTables:
    tgtCursor.execute("DROP TABLE IF EXISTS %s" % t[0])
    srcDataset = etl.fromdb(srcConnections,"SELECT * FROM %s" % t[0])
    etl.todb(srcDataset,tgtConnection,t[0],create = True, sample=1000)
    
