from flask import Flask, render_template, Response, request, redirect, url_for
from app.database import database
import random
from pprint import pprint
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statistics
from pyclustering.cluster.kmedoids import kmedoids as kmedoids2
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.elbow import elbow as elbow2
from pyclustering.cluster.center_initializer import random_center_initializer
from sklearn.datasets import make_blobs
from random import randrange, uniform
from datetime import datetime


app = Flask(__name__)

database = database()

@app.route('/dashboard')
def dashboard():
    jumlah_data = len(database.getData())
    # silhouete = database.getAllSilhoute()

    return render_template('home/dashboard.html',
        jumlah = jumlah_data,
    )

@app.route('/data')
def data():
    data = database.getData()
    jumlah = len(data)

    return render_template('data/list_data.html', jumlah = jumlah, data=data)

@app.route('/data_cluster/<cluster>')
def data_cluster(cluster):
    data = database.getDatabyCluster(cluster)
    jumlah = len(data)

    return render_template('data/list_dataCluster.html', jumlah = jumlah, data=data, cluster=cluster)

@app.route('/list_kualitas')
def list_kualitas():
    data = database.getData()
    jumlah = len(data)

    return render_template('data/list_kualitas.html', jumlah = jumlah, data=data)

@app.route('/elbow')
def elbow():
    data = database.getData()
    data = pd.DataFrame(data)

    id_data = data['id_data'].values
    data_normalization = np.array(list(zip(data['pci_kanan'], data['pci_kiri'], data['iri_kanan'], data['iri_kiri'])))
    kmin, kmax = 1, 10
    elbow_instance = elbow2(data_normalization, kmin, kmax, initializer=random_center_initializer)
    elbow_instance.process()
    elbow_score = elbow_instance.get_wce()

    min_ = 9999
    max_ = -1

    for data in elbow_score:
        if data > max_:
            max_ = data
        if data < min_:
            min_ = data

    for i in range(0, len(elbow_score)):
        elbow_score[i] = (elbow_score[i]-min_)/(max_-min_)

    plt.plot(elbow_score)
    plt.title("Elbow Diagram")
    plt.savefig("static/elbow.jpg")
    plt.close()

    return render_template('cluster/elbow.html', elbow=elbow_score)

@app.route('/kmedoidsindex')
def kmedoidsindex():
    return render_template('cluster/kmedoids.html')

@app.route('/silhoute_index')
def silhoute_index():
    return render_template('cluster/silhouete_index.html')

@app.route('/kmedoids', methods=['POST'])
def kmedoids():
    # distance = request.values.get('distance')
    k = request.values.get('k')

    data = database.getData()
    data = pd.DataFrame(data)

    normalisasi = []
    kolom = ['pci_kiri', 'pci_kanan', 'iri_kiri', 'iri_kanan']
    for i,x in enumerate(kolom):
        normalisasi.append([max(data[x]),min(data[x])])

    data_normal = []
    for i,x in enumerate(kolom):
        data_normal.append((data[x].values-normalisasi[i][1])/(normalisasi[i][0] - normalisasi[i][1]))

    id_data = data['id_data'].values
    data_training = np.array(list(zip(data_normal[0],data_normal[1],data_normal[2],data_normal[3])))
    data_medoids = np.array(list(zip(data['pci_kanan'].values,data['pci_kiri'].values,data['iri_kanan'].values,data['iri_kiri'].values)))
    ganti_data = np.array(list(zip(id_data)))

    medoid_awal = list()
    i = 40
    rata_rata = len(data) / int(k)
    for x in range(int(k)):
        medoid_awal.append(i)
        i = i + int(rata_rata)
    
    kmedoid = kmedoids2(data_training, medoid_awal);
    kmedoid.process()
    clusters_list = kmedoid.get_clusters()
    medoids = kmedoid.get_medoids()

    cdict = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray', 9: 'olive', 10: 'cyan'}
    ax = plt.axes(projection='3d')
    plt.figsize=(10,5)
    plt.title('Visualisasi')
    i = 1
    for c_,m in zip(clusters_list,medoids):
        ax.scatter3D(data_training[c_,1],data_training[c_,2],data_training[c_,3],c=data_training[c_,1], label='Cluster '+str(i))

        for tmp in c_ :
            database.setCluster(ganti_data[tmp][0], i)
        i = i + 1

    for c_,m in zip(clusters_list,medoids):
        ax.scatter3D(data_training[m, 1], data_training[m, 2], data_training[m, 3], marker='x',c='black',label='Cluster')

    plt.savefig("static/3d_display1.jpg")
    plt.close()

    ax = plt.axes(projection='3d')
    plt.figsize=(10,5)
    plt.title('Visualisasi')
    i = 1
    list_medoids = []
    for c_,m in zip(clusters_list,medoids):
        ax.scatter3D(data_training[c_,1],data_training[c_,2],data_training[c_,3], label='Cluster '+str(i))
        list_medoids.append(data_medoids[m])

        for tmp in c_ :
            database.setCluster(ganti_data[tmp][0], i)
        i = i + 1

    for c_,m in zip(clusters_list,medoids):
        ax.scatter3D(data_training[m, 1], data_training[m, 2], data_training[m, 3], marker='o',c='black')

    ax.legend(loc=1)
    plt.savefig("static/3d_display2.jpg")
    plt.close()

    return render_template('cluster/hasil_kmedoids.html', k=int(k), medoids = list_medoids)

@app.route('/silhouete', methods=['POST'])
def silhouete():
    k = request.values.get('k')
    silhouete_score = []

    data = database.getData()
    data = pd.DataFrame(data)

    normalisasi = []
    kolom = ['pci_kiri', 'pci_kanan', 'iri_kiri', 'iri_kanan']
    initial_centroid = 10
    for i,x in enumerate(kolom):
        normalisasi.append([max(data[x]),min(data[x])])

    data_normal = []
    for i,x in enumerate(kolom):
        data_normal.append((data[x].values-normalisasi[i][1])/(normalisasi[i][0] - normalisasi[i][1]))

    id_data = data['id_data'].values
    data_training = np.array(list(zip(data_normal[0],data_normal[1],data_normal[2],data_normal[3])))
    ganti_data = np.array(list(zip(id_data)))

    medoid_awal = list()
    i = 40
    rata_rata = len(data) / int(k)
    for x in range(int(k)):
        medoid_awal.append(i)
        i = i + int(rata_rata)
    
    kmedoid = kmedoids2(data_training, medoid_awal);
    kmedoid.process()
    clusters_list = kmedoid.get_clusters()
    medoids = kmedoid.get_medoids()

    silhoutte_score = silhouette(data_training, clusters_list).process().get_score()
    score = []
    jumlah = 0

    for i in range(0, int(k)):
        print(k)
        score.append(initial_centroid)
        data_cluster = database.getDatabyCluster(i+1)
        for tmp in range(0,len(data_cluster)):
            score[i] = score[i] + silhoutte_score[tmp]

        score[i] = score[i] / (len(data_cluster))
        jumlah = jumlah + score[i]        
    jumlah = jumlah / int(k)

    return render_template('cluster/silhouete.html', silhouete_score = score, jumlah = jumlah,k=k)

@app.route('/tambah_index')
def tambah_index():
    return render_template('data/tambah.html')

@app.route('/edit_index/<id_data>')
def edit_index(id_data):
    data = database.getDatabyid(id_data)[0];

    return render_template('data/edit.html', data = data)

@app.route('/proses_tambah', methods=['GET', 'POST'])
def proses_tambah():
    nama = request.values.get('nama')
    tgl_mulai = request.values.get('tgl_mulai')
    tgl_selesai = request.values.get('tgl_selesai')
    R = request.values.get('R')
    F = request.values.get('F')
    M = request.values.get('M')

    tgl_mulai = datetime.strptime(tgl_mulai, '%Y-%m-%d')
    tgl_selesai = datetime.strptime(tgl_selesai, '%Y-%m-%d')

    new_data = (nama,tgl_mulai,tgl_selesai,R,F,M,0)

    database.insert(new_data)

    return redirect(url_for('data'))

@app.route('/kesimpulan')
def kesimpulan():
    data_pci = []
    for i in range(1,4):
        data_pci.append(hitungKesimpulan(database.getDatabyCluster(i), "pci"))

    i = 1
    for j, row in enumerate(data_pci):
        # data[j][3] = (row[2]*100)+(row[1]*10)+(row[0]*0)
        data_pci[j][3] = row[2]
        data_pci[j][4] = i
        i = i + 1

    n = len(data_pci)
    for x in range(n):
        for z in range(0, n-x-1):
            if data_pci[z][3] > data_pci[z+1][3]:
                tmp = deepcopy(data_pci[z])
                data_pci[z] = deepcopy(data_pci[z+1])
                data_pci[z+1] = deepcopy(tmp)    
    
    data_iri = []
    for i in range(1,4):
        data_iri.append(hitungKesimpulan(database.getDatabyCluster(i), "iri"))

    i = 1
    for j, row in enumerate(data_iri):
        # data[j][3] = (row[2]*100)+(row[1]*10)+(row[0]*0)
        data_iri[j][3] = row[2]
        data_iri[j][4] = i
        i = i + 1

    n = len(data_iri)
    for x in range(n):
        for z in range(0, n-x-1):
            if data_iri[z][3] > data_iri[z+1][3]:
                tmp = deepcopy(data_iri[z])
                data_iri[z] = deepcopy(data_iri[z+1])
                data_iri[z+1] = deepcopy(tmp)

    return render_template('cluster/kesimpulan.html', data_pci = data_pci, data_iri = data_iri)

@app.route('/proses_edit', methods=['GET', 'POST'])
def proses_edit():
    id_data = request.values.get('id_data')
    nama = request.values.get('nama')
    tgl_mulai = request.values.get('tgl_mulai')
    tgl_selesai = request.values.get('tgl_selesai')
    R = request.values.get('R')
    F = request.values.get('F')
    M = request.values.get('M')

    tgl_mulai = datetime.strptime(tgl_mulai, '%Y-%m-%d')
    tgl_selesai = datetime.strptime(tgl_selesai, '%Y-%m-%d')

    new_data = (nama,tgl_mulai,tgl_selesai,R,F,M)

    database.update_data(id_data, new_data)

    return redirect(url_for('data'))

@app.route('/indeximport')
def indeximport():
    pass

@app.route('/proses_import', methods=['GET', 'POST'])
def proses_import():
    file = request.files['file']
    data = pd.read_csv(file)
    
    for x in range(data.shape[0]):
        c = data.iloc[x]
        new_data = (c[0],c[1],c[2],c[3],c[4],c[5],c[6])
        database.insert(new_data)

    return redirect(url_for('data'))

@app.route('/proses_delete/<id_data>')
def proses_delete(id_data):
    database.delete(id_data)
    return redirect(url_for('data'))

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def indic(data):
    #tampilan purpose

    max = (np.max(data, axis=1))
    min = (np.min(data, axis=1))

    max = [x - random.uniform(0,1) for x in max]
    min = [x + random.uniform(0,1) for x in min]

    return max, min

def average(list):
    return sum(list)/len(list)

def deleteList(data, index):
     data = data[:index] + data[index+1:]

     return data

def hitungKesimpulan(data_cluster, type):
    kolom_pci = ['pci_kanan', 'pci_kiri']
    kolom_iri = ['iri_kanan', 'iri_kiri']

    matrix = [0, 0, 0, 0, 0]
    for data in data_cluster:
        if type == "pci":
            for kolom in kolom_pci:
                if data[kolom] >= 85:
                    matrix[0] = matrix[0] + 1
                elif data[kolom] >= 70 and data[kolom] < 85:
                    matrix[1] = matrix[1] + 1
                elif data[kolom] < 70 :
                    matrix[2] = matrix[2] + 1
        else:
            for kolom in kolom_iri:
                if data[kolom] < 4:
                    matrix[0] = matrix[0] + 1
                elif data[kolom] >= 4 and data[kolom] <= 8:
                    matrix[1] = matrix[1] + 1
                elif data[kolom] > 8:
                    matrix[2] = matrix[2] + 1

    return matrix

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='127.0.0.1',port='8000', debug=True)
    # kesimpulan()