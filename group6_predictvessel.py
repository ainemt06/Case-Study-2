import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
import functools
from sklearn.metrics.cluster import adjusted_rand_score

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred_dict = predictor(csv_path)
    
    best_score = -1
    best_model = None
    for model_name, labels_pred in labels_pred_dict.items():
        score = adjusted_rand_score(labels_true, labels_pred)
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f'Best Adjusted Rand Index Score: {best_score:.4f} from model: {best_model}')


# Affinity Propagation - chooses number of clusters based on "exemplar" points
def affinity_propagation(X):
    model = AffinityPropagation(random_state=123)
    return model.fit_predict(X)

# Agglomerative Clustering - builds a hierarchy of clusters
def agglomerative_clustering(X):
    model = AgglomerativeClustering(distance_threshold=1.0, linkage='ward') # refine
    return model.fit_predict(X)

# DBSCAN - groups together points that are close to each other
def dbscan(X):
    model = DBSCAN(eps=0.5, min_samples=5)
    return model.fit_predict(X)

def predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)

    labels_pred = {
        'affinity_propagation': affinity_propagation(X),
        'agglomerative_clustering': agglomerative_clustering(X),
        'dbscan': dbscan(X)
    }

    return labels_pred




if __name__=="__main__":
    get_baseline_score()
    evaluate()


