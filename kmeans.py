'''
k-means trace clustering (random or advanced initialization) using :
- Levenshtein distance
- Augmenteed Levenshtein distance
- Euclidean distance
- Augmented Levenshtein distance

@returns:
Internal validation:
- Silhoutte Coefficient
- Calinski-Harabsz Index
- Davies-Bouldin Index
Process model evaluation using alpha, heuristic, or inductive miner algorithm:
- Fitness
- Precision
- Generalization 
- Simplicity 
'''


import time
import random
import sys
import numpy as np
import pandas
from numpy.core.numeric import NaN
from distance import extended_euclidean, euclidean, extended_levenshtein, levenshtein
from evaluation import insintric, process_models, modelevaluation, average
import pm4py

# Distance measure can either be 'Euclidean' or 'Levenshtein'
DISTANCE_MEASURE = 'Levenshtein'

# Use the selected distance measure in its traditional or augmented form
AUGMENTED = True

# Set number of clusters
K = 3

# If set False, initial centroids will be selected randomly
ADVANCED_INITIALIZATION = False

# Discovery algorithm can be 'alpha', 'heuristic', or 'inductive'
MINER = 'inductive'

# Filetyp can be 'xes' or 'csv'
FILETYPE = 'xes'

LOGPATH = "logs/variants/HospitalBilling_sampled_variants.xes"
AUGMENTED_LOGPATH = "logs/augmented/HospitalBilling_sampled_variants_augmented.csv"


if FILETYPE == 'xes':
    #Event log as xes-file
    LOG = pm4py.read_xes(LOGPATH)
    EVENTLOG = pm4py.convert_to_dataframe(LOG)
else:
    #Event log as csv-file
    LOG = pandas.read_csv(LOGPATH, sep=';')
    EVENTLOG = pandas.read_csv(LOGPATH, sep=';')


#Prepare logs for clustering
EVENTLOG_AUGMENTED = pandas.read_csv(AUGMENTED_LOGPATH, sep=';')
EVENTLOG_AUGMENTED = EVENTLOG_AUGMENTED.rename(columns={'case:concept:name': 'case_id', 'object:name': 'business_object', 'action:name': 'action', 'action:status': 'status'})
EVENTLOG_AUGMENTED = EVENTLOG_AUGMENTED.replace(to_replace=NaN, value='none')
EVENTLOG = EVENTLOG.rename(columns={'case:concept:name': 'case_id', 'concept:name':'event'})
EVENTLOG = EVENTLOG.replace(to_replace=NaN, value='none')

#Changes the id of traces
def changeCaseId():
    i = 1
    for id in EVENTLOG.case_id.unique():
        EVENTLOG.loc[EVENTLOG.case_id == id, 'case_id'] = i
        i+=1

#Defines the stopping criterion of the clustering process
def stoppingCriterion(oldCentroids, newCentroids):
    bool = True
    if len(oldCentroids) != len(newCentroids):
        bool = False
    for oldC in oldCentroids:
        if oldC not in newCentroids:
            bool = False
    return bool

#The actual k-means clustering algorithm
def kmeansClustering():
    #Transform traces
    print('Start kmeans Clustering:')
    if AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean':
        dicts, objects, actions = extended_euclidean.createDicts(EVENTLOG_AUGMENTED)
    elif AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean':
        dicts, events = euclidean.createDicts(EVENTLOG)
    elif AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
        object_dicts, action_dicts = extended_levenshtein.createDicts(EVENTLOG_AUGMENTED)
    elif AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein':
        dicts = levenshtein.createDicts(EVENTLOG)

    clusters = []

    if AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
        object_centroids = []
        action_centroids = []
        distances = np.zeros((K,len(object_dicts)))
        number = len(object_dicts)
    else:
        centroids = []
        distances = np.zeros((K,len(dicts)))
        number = len(dicts)

    #Initialization of centroids
    print('Finding initial clusters...')
    if ADVANCED_INITIALIZATION == False:
        print('...randomly')
        randomlist = random.sample(range(1,number),K)
        print(randomlist)

        if AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
            for trace in object_dicts:
                if trace['id'] in randomlist:
                    object_centroids.append(trace)
            for trace in action_dicts:
                if trace['id'] in randomlist:
                    action_centroids.append(trace)
        else:
            for trace in dicts:
                if trace['id'] in randomlist:
                    centroids.append(trace)

    elif ADVANCED_INITIALIZATION == True:
        print('...selectively')
        randomlist = []
        randomlist.append(random.choice(range(1,number)))
        while len(randomlist) <= K:
            same = False
            randomNo = random.choice(range(1,number))
            matrix = np.zeros((len(randomlist),1))
            randomTrace = []
            randomObject = []
            randomAction = []
            centroids = []
            object_centroids = []
            action_centroids = []

            if AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
                for trace in object_dicts:
                    if trace['id'] in randomlist:
                        object_centroids.append(trace)
                    elif trace['id']==randomNo:
                        randomObject.append(trace)
                for trace in action_dicts:
                    if trace['id'] in randomlist:
                        action_centroids.append(trace)
                    elif trace['id']==randomNo:
                        randomAction.append(trace)
            else:
                for trace in dicts:
                    if trace['id'] in randomlist:
                        centroids.append(trace)
                    elif trace['id']==randomNo:
                        randomTrace.append(trace)

            if len(randomlist) == K:
                break

            if AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean':
                matrix = extended_euclidean.computeDistanceMatrix(centroids, randomTrace, objects)
            elif AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean':
                matrix = euclidean.computeDistanceMatrix(centroids, randomTrace)
            elif AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
                matrix = extended_levenshtein.computeDistanceMatrix(object_centroids, action_centroids, randomObject, randomAction)
            elif AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein':
                matrix = levenshtein.computeDistanceMatrix(centroids, randomTrace)

            for j in range(0, len(randomlist)):
                try:
                    if matrix[j][0] <= 1:
                        same = True
                except Exception:
                    same = True
                    randomlist = []
                    randomlist.append(random.choice(range(1,number)))
            if same == True:
                print('---redo---')
            else:
                randomlist.append(randomNo)
                print(randomlist)

    print('Starting clustering process...')
    i = 1
    #Max 8 iterations
    while i < 8:
        print('Iteration {}:'.format(i))
        clusters = []
        labels = []
        for no in range(1,K+1):
            cluster_dict = {'id':no, 'traces':[]}
            clusters.append(cluster_dict)

        #1. Computes distances
        print('Calculating distances...')
        if AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean':
            distances = extended_euclidean.computeDistanceMatrix(centroids, dicts, objects)
        elif AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean':
            distances = euclidean.computeDistanceMatrix(centroids, dicts)
        elif AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
            distances = extended_levenshtein.computeDistanceMatrix(object_centroids, action_centroids, object_dicts, action_dicts)
        elif AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein':
            distances = levenshtein.computeDistanceMatrix(centroids, dicts)

        #2. Selects fitting centroid for each trace
        print('Selecting correct cluster...')
        for d2 in range(0,len(distances[0])):
            min_distance = 1000
            possibleClusters = []
            clusterNo = 0
            for d1 in range(0,K):
                if distances[d1][d2] == min_distance:
                    possibleClusters.append(d1+1)
                if distances[d1][d2] < min_distance:
                    min_distance = distances[d1][d2]
                    possibleClusters = []
                    possibleClusters.append(d1+1)
            clusterNo = random.choice(possibleClusters)
            labels.append(clusterNo)
            for c in clusters:
                if c['id']==clusterNo:
                    c['traces'].append(d2+1)
    
        #3. Computes new centroid
        print('Computing new centroids...')
        if AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean':
            newCentroids = []
            for c in clusters:
                traces = []
                for d in dicts:
                    if d['id'] in c['traces']:
                        traces.append(d)
                newCentroids.append(extended_euclidean.computeMean(traces, objects, actions, c['id']))

        elif AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean':
            newCentroids = []
            for c in clusters:
                traces = []
                for d in dicts:
                    if d['id'] in c['traces']:
                        traces.append(d)
                newCentroids.append(euclidean.computeMean(traces, events, c['id']))

        elif AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
            newCentroids_objects = []
            newCentroids_actions = []
            for c in clusters:
                object_traces = []
                action_traces = []
                for d in object_dicts:
                    if d['id'] in c['traces']:
                        object_traces.append(d)
                for d in action_dicts:
                    if d['id'] in c['traces']:
                        action_traces.append(d)
                oc, ac = extended_levenshtein.computeCentroid(object_traces, action_traces, c['id'])
                newCentroids_objects.append(oc)
                newCentroids_actions.append(ac)

        elif AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein':
            newCentroids = []
            for c in clusters:
                traces = []
                for d in dicts:
                    if d['id'] in c['traces']:
                        traces.append(d)
                newCentroids.append(levenshtein.computeCentroid(traces, c['id']))

        #4. Stopping Criterion
        print('Testing for stopping criterion...')
        if (AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean') or (AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean') or (AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein'):
            if stoppingCriterion(centroids, newCentroids) == True:
                print('Stopp')
                break
            else:
                centroids = newCentroids
                i+=1
                print('Next iteration...')
        else:
            if stoppingCriterion(object_centroids, newCentroids_objects) == True and stoppingCriterion(action_centroids, newCentroids_actions)==True:
                print('Stopp')
                break
            else:
                object_centroids = newCentroids_objects
                action_centroids = newCentroids_actions
                i+=1
                print('Next iteration...')

    #Computes complete distance matrix between all traces
    if AUGMENTED == True and DISTANCE_MEASURE == 'Euclidean':
        X = extended_euclidean.computeDistanceMatrix(dicts, dicts, objects)
    elif AUGMENTED == False and DISTANCE_MEASURE == 'Euclidean':
        X = euclidean.computeDistanceMatrix(dicts, dicts)
    elif AUGMENTED == True and DISTANCE_MEASURE == 'Levenshtein':
        X = extended_levenshtein.computeDistanceMatrix(object_dicts, action_dicts, object_dicts, action_dicts)
    elif AUGMENTED == False and DISTANCE_MEASURE == 'Levenshtein':
        X = levenshtein.computeDistanceMatrix(dicts, dicts)

    print('Finished within {} iterations.'.format(i))
    print("__________________________________________________")
    print()
    return X, labels

def main():
    main_tic = time.perf_counter()
    print("__________________________________________________")
    print()

    X, labels = kmeansClustering()
    main_toc = time.perf_counter()
    print(f"Clustering took {main_toc - main_tic:0.4f} seconds")
    print("__________________________________________________")
    print()

    #Internal validation
    siScore, chScore, dbScore = insintric.evaluateClusters(X, labels)

    if AUGMENTED:
        print("kMeans-Clustering using the augmented {} distance:".format(DISTANCE_MEASURE))
    else:
        print("kMeans-Clustering using the {} distance:".format(DISTANCE_MEASURE))

    print("Silhoutte Coefficient = {}".format(siScore))
    print("Calinski-Harabsz Index = {}".format(chScore))
    print("Davies-Bouldin Index = {}".format(dbScore))
    print("__________________________________________________")
    print()

    #Process model evaluation of the whole event log
    if MINER == 'alpha':
        net, im, fm = process_models.alphaMiner(LOG)
    elif MINER == 'heuristic':
        net, im, fm = process_models.heuristicMiner(LOG)
    elif MINER == 'inductive':
        net, im, fm = process_models.inductiveMiner(LOG)

    fitness_token = modelevaluation.getReplayFitness(LOG, net, im, fm, False)
    precision_token = modelevaluation.getPrecision(LOG, net, im, fm, False)
    generalization = modelevaluation.getGeneralization(LOG, net, im, fm)
    simplicity = modelevaluation.getSimplictiy(net)

    print("The process model of the whole event log using '{} Miner' achieves the following scores".format(MINER))
    print("Fitness (token): {}".format(fitness_token))
    print("Precision (token): {}".format(precision_token))
    print("Generalization: {}".format(generalization))
    print("Simplicity: {}".format(simplicity))
    print("__________________________________________________")
    print()

    changeCaseId()

    #Process model evaluation of each cluster log
    for i in range(1,K+1):
        cluster_log = process_models.getClusterLog(labels, EVENTLOG, i)
        if len(cluster_log) > 0:
            print(len(cluster_log))
            if MINER == 'alpha':
                net, im, fm = process_models.alphaMiner(cluster_log)
            elif MINER == 'heuristic':
                net, im, fm = process_models.heuristicMiner(cluster_log)
            elif MINER == 'inductive':
                net, im, fm = process_models.inductiveMiner(cluster_log)

            fitness_token = modelevaluation.getReplayFitness(cluster_log, net, im, fm, False)
            precision_token = modelevaluation.getPrecision(cluster_log, net, im, fm, False)
            generalization = modelevaluation.getGeneralization(cluster_log, net, im, fm)
            simplicity = modelevaluation.getSimplictiy(net)

            print("The process models of each trace {} using '{} Miner' achieve the following scores".format(i, MINER))
            print("Fitness (token): {}".format(fitness_token))
            print("Precision (token): {}".format(precision_token))
            print("Generalization: {}".format(generalization))
            print("Simplicity: {}".format(simplicity))
            print("__________________________________________________")
            print()

    avgD = average.averageDistance(X, len(LOG))
    print("Average distance = {}".format(avgD))

if __name__ == '__main__':
    maintic = time.perf_counter()
    main()
    maintoc = time.perf_counter()
    print(f"Program finished all operations in {maintoc - maintic:0.4f} seconds")
    sys.exit()