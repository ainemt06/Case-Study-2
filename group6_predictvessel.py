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
    model = AffinityPropagation(damping=.9, max_iter=200, convergence_iter=15, random_state=123)
    return model.fit_predict(X)

# Agglomerative Clustering - builds a hierarchy of clusters
def agglomerative_clustering(X):
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward') # refine
    return model.fit_predict(X)

# DBSCAN - groups together points that are close to each other
def dbscan(X):
    model = DBSCAN(eps=.5, min_samples=5)
    return model.fit_predict(X)

def predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)

    aprop = affinity_propagation(X)
    agg = agglomerative_clustering(X)
    db = dbscan(X)

    print("Predicted number of clusters for affinity propagation:", len(np.unique(aprop)))
    print("Predicted number of clusters for agglomerative clustering:", len(np.unique(agg)))
    print("Predicted number of clusters for DBSCAN:", len(np.unique(db)))

    labels_pred = {
        'affinity_propagation': aprop,
        'agglomerative_clustering': agg,
        'dbscan': db
    }

    return labels_pred


if __name__=="__main__":
    get_baseline_score()
    evaluate()


