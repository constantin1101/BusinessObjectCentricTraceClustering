from io import StringIO
from numpy.core.numeric import NaN
import numpy as np
import pandas

#Transforms traces into list of events
def createDicts(event_log):
    events = event_log.event.unique()
    print(events)

    dicts = []

    i = 1
    for id in event_log.case_id.unique() :
        dict_trace = {"id": i}
        trace = event_log[event_log["case_id"]==id]
        event_list = []
        for event in trace["event"]:
            event_list.append(event)
        dict_trace['list']=event_list
        dicts.append(dict_trace)
        i += 1

    return dicts

#Dynamic programming approach: Iterative with full matrix 
# "https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix"
def levenstheinDistance(first, second):
    matrix = np.zeros((len(first)+1,len(second)+1))

    for i in range(len(first)+1):
        matrix[i, 0] = i
    
    for j in range(len(second)+1):
        matrix[0, j] = j
    
    for i in range(1,len(first)+1): 
        for j in range(1,len(second)+1): 
            if first[i-1] == second[j-1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            
            matrix[i,j] = min(matrix[i][j-1] + 1,  
                                matrix[i-1][j] + 1,        
                                matrix[i-1][j-1] + substitutionCost)

    return matrix[len(first)][len(second)]

#Computes the distance matrix between centroids and traces
def computeDistanceMatrix(centroids, dicts):
    matrix = np.zeros((len(centroids),len(dicts)))
    i=0
    for c in centroids:
        j=0
        for d in dicts:
            first = np.asarray(c['list'])
            second = np.asarray(d['list'])
            distance = levenstheinDistance(first, second)
            matrix[i][j] = distance
            j+=1
        i+=1
    return matrix

#Coputes the new centroid for a cluster
def computeCentroid(traces, no):
    centroid = {'id':'centroid{}'.format(no)}
    matrix = computeDistanceMatrix(traces, traces)
    min = 100000
    id = 0

    for i in range(0, len(traces)):
        sum = 0
        for j in range(0, len(traces)):
            sum += matrix[i][j]
        average = sum/len(traces)
        if average < min:
            min = average
            id = i

    centroid['list'] = traces[id]['list']
    return centroid
