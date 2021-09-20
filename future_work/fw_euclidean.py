from numpy.core.numeric import NaN
import numpy as np

#Transforms traces into 'business-object' vectors
def createDicts(event_log):
    num_cases = len(event_log.case_id.unique())
    print('Traces: {}'.format(num_cases))

    objects = event_log.business_object.unique()
    actions = event_log.action.unique()
    actors = event_log.actor.unique()

    print(num_cases)
    print('Unique BO: {}'.format(len(objects)))
    print('Unique Actions: {}'.format(len(actions)))
    print('Unique Actors: {}'.format(len(actors)))

    dicts = []

    i = 1
    for id in event_log.case_id.unique() :
        dict_trace = {"id": i}
        trace = event_log[event_log["case_id"]==id]
        for obj in objects:
            object_vector = []
            for action in actions:
                object_vector.append(len(trace[(trace['business_object']==obj) & (trace['action']==action)]))
            for actor in actors:
                object_vector.append(len(trace[(trace['business_object']==obj) & (trace['actor']==actor)]))
            dict_trace[obj]=object_vector
        dicts.append(dict_trace)
        i+=1

    return dicts, objects, actions, actors

#Computes the distance matrix between centroids and traces
def computeDistanceMatrix(centroids, dicts, objects):
    matrix = np.zeros((len(centroids),len(dicts)), dtype=np.float)
    i=0
    for c1 in centroids:
        j=0
        for d2 in dicts:
            distances = []
            for obj in objects:
                point1 = np.asarray(c1[obj])
                point2 = np.asarray(d2[obj])
                #Euclidean distance
                distances.append(np.linalg.norm(point1 - point2))
            sum = 0
            for d in distances:
                sum = sum + d
            matrix[i][j] = sum
            j+=1
        i+=1
    return matrix

def computeDistanceMatrix_new(centroids, dicts, objects):
    matrix = np.zeros((len(centroids),len(dicts)), dtype=np.float)
    i=0
    for c1 in centroids:
        j=0
        for d2 in dicts:
            sum = 0
            for obj in objects:
                point1 = np.asarray(c1[obj])
                point2 = np.asarray(d2[obj])
                sum = sum + (np.linalg.norm(point1 - point2))
            matrix[i][j] = sum
            j+=1
        i+=1
    return matrix

#Computes the new centroid for a cluster
def computeMean(traces, objects, actions, actors, no):
    mean = {'id':'centroid{}'.format(no)}
    for obj in objects:
        object_vector = []
        size = len(actions) + len(actors)
        for n in range(0, size):
            sum = 0
            if len(traces) != 0:
                for trace in traces:
                    sum += trace[obj][n]
                object_vector.append(sum/len(traces))
            else:
                object_vector.append(0)
        mean[obj]=object_vector
    return mean