import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN, OPTICS
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
    csv_path = './Data/set1.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred_dict = predictor(csv_path, labels_true)
    
    best_score = -1
    best_model = None
    for model_name, labels_pred in labels_pred_dict.items():
        score = adjusted_rand_score(labels_true, labels_pred)
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f'Best Adjusted Rand Index Score: {best_score:.4f} from model: {best_model}')


# Affinity Propagation - chooses number of clusters based on "exemplar" points
def affinity_propagation(X, labels_true=None):
    best_score = -1
    best_labels = None
    best_params = None
    preferences = [None, -50, -40, -30, -20]
    dampings = [0.9]

    for damping in dampings:
        for preference in preferences:
            model = AffinityPropagation(
                damping=damping,
                max_iter=300,
                convergence_iter=20,
                preference=preference,
                random_state=123
            )
            labels = model.fit_predict(X)
            if labels_true is None:
                return labels
            score = adjusted_rand_score(labels_true, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = {'damping': damping, 'preference': preference}

    print(f'AffinityPropagation best ARI {best_score:.4f} with {best_params}')
    return best_labels

# Agglomerative Clustering - builds a hierarchy of clusters
def agglomerative_clustering(X, labels_true=None):
    best_score = -1
    best_labels = None
    best_params = None
    distance_thresholds = [25, 30, 35, 40, 45, 50]

    for distance_threshold in distance_thresholds:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward')
        labels = model.fit_predict(X)
        if labels_true is None:
            return labels
        score = adjusted_rand_score(labels_true, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_params = {'linkage': 'ward', 'distance_threshold': distance_threshold}

    print(f'AgglomerativeClustering best ARI {best_score:.4f} with {best_params}')
    return best_labels

# DBSCAN - groups together points that are close to each other
def dbscan(X, labels_true=None):
    best_score = -1
    best_labels = None
    best_params = None
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    min_samples_values = [3, 5, 8, 10]

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            if labels_true is None:
                return labels
            score = adjusted_rand_score(labels_true, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = {'eps': eps, 'min_samples': min_samples}

    print(f'DBSCAN best ARI {best_score:.4f} with {best_params}')
    return best_labels

# OPTICS - density-based clustering for variable density
def optics(X, labels_true=None):
    best_score = -1
    best_labels = None
    best_params = None
    min_samples_values = [5, 10, 15]
    max_eps_values = [np.inf, 2.0, 3.0]

    for min_samples in min_samples_values:
        for max_eps in max_eps_values:
            model = OPTICS(min_samples=min_samples, max_eps=max_eps)
            labels = model.fit_predict(X)
            if labels_true is None:
                return labels
            score = adjusted_rand_score(labels_true, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = {'min_samples': min_samples, 'max_eps': max_eps}

    print(f'OPTICS best ARI {best_score:.4f} with {best_params}')
    return best_labels

def predictor(csv_path, labels_true=None):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)

    #aprop = affinity_propagation(X, labels_true)
    agg = agglomerative_clustering(X, labels_true)
    db = dbscan(X, labels_true)
    #opt = optics(X, labels_true)

    #print("Predicted number of clusters for affinity propagation:", len(np.unique(aprop)))
    print("Predicted number of clusters for agglomerative clustering:", len(np.unique(agg)))
    print("Predicted number of clusters for DBSCAN:", len(np.unique(db)))

    labels_pred = {
        #'affinity_propagation': aprop,
        'agglomerative_clustering': agg,
        'dbscan': db,
        #'optics': opt
    }

    return labels_pred


if __name__=="__main__":
    get_baseline_score()
    evaluate()


