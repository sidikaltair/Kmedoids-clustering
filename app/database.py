import pymysql.cursors, os
from pprint import pprint

class database():
    def __init__(self):
        print("init")

    def openDb(self):
        global conn, cursor
        conn = pymysql.connect(host='localhost',user='root',password='',database='db_siabah',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)

        cursor = conn.cursor()   

    def closeDb(self):
        global conn, cursor
        cursor.close()
        conn.close()

    def insert(self, data):
        self.openDb()
        container = []
        sql = "INSERT INTO data (nama,tgl_mulai,tgl_selesai,R,F,M,cluster) VALUES(%s,%s,%s,%s,%s,%s,%s)";
        cursor.execute(sql, data)
        conn.commit()
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def delete(self, id_data):
        self.openDb()
        container = []
        sql = "DELETE from data WHERE id_data = "+id_data;
        cursor.execute(sql)
        conn.commit()
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def getData(self):
        self.openDb()
        container = []
        sql = "SELECT * FROM data"
        cursor.execute(sql)
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def getDatabyid(self, id_data):
        self.openDb()
        container = []
        sql = "SELECT * FROM data where id_data = "+id_data;
        cursor.execute(sql)
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def getDatabyCluster(self, cluster):
        self.openDb()
        container = []
        sql = "SELECT * FROM data where cluster = "+str(cluster)
        cursor.execute(sql)
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def setCluster(self, id_data, value):
        self.openDb()
        sql = "UPDATE data SET cluster = %s WHERE id_data = " + str(id_data)
        cursor.execute(sql, value)
        conn.commit()
        self.closeDb()
        return "update done"

    def ambilSemuaSilhoute(self):
        self.openDb()
        container = []
        sql = "SELECT * FROM silhoute";
        cursor.execute(sql)
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def ambilSilhoute(self, distance, k):
        self.openDb()
        container = []
        sql = "SELECT * FROM silhoute where distance = "+str(distance)+" and k = "+str(k);
        cursor.execute(sql)
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container        

    def updateSilhoute(self, id_silhoute, data):
        self.openDb()
        sql = "UPDATE silhoute SET distance=%s,k=%s,nilai=%s WHERE id_silhoute = " + str(id_silhoute)
        cursor.execute(sql, data)
        conn.commit()
        self.closeDb()
        return "update done"

    def tambahSilhoute(self, data):
        self.openDb()
        container = []
        sql = "INSERT INTO silhoute (distance,k,nilai) VALUES(%s,%s,%s)";
        cursor.execute(sql, data)
        conn.commit()
        results = cursor.fetchall()
        for data in results:
          container.append(data)
        self.closeDb()
        return container

    def update_data(self, id_data, value):
        self.openDb()
        sql = "UPDATE data SET nama=%s,tgl_mulai=%s,tgl_selesai=%s,R=%s,F=%s,M=%s WHERE id_data = " + str(id_data)
        cursor.execute(sql, value)
        conn.commit()
        self.closeDb()
        return "update done"

    def csv_reader():
        reader_csv = csv.reader(open('data/datatest.csv'))

        for line in reader_csv:
            training_obs_csv.append(line[:-1])
            training_cat_csv.append(line[-1])