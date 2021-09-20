import numpy as np

#Transforms traces into vectors (bags-of-activities)
def createDicts(event_log):
    events = event_log.event.unique()
    print('Unique events: {}'.format(len(events)))
    dicts = []

    i = 1
    for id in event_log.case_id.unique() :
        dict_trace = {"id": i}
        trace = event_log[event_log["case_id"]==id]
        event_vector = []
        for event in events:
            event_vector.append(len(trace[trace['event']==event]))
        dict_trace['vector']=event_vector
        '''add = True
        for d in dicts:
            if d['vector']==event_vector:
                add = False
                break
        if add == True:###'''
        dicts.append(dict_trace)
        i += 1

    return dicts, events

#Computes the distance matrix between centroids and traces
def computeDistanceMatrix(centroids, dicts):
    matrix = np.zeros((len(centroids),len(dicts)), dtype=np.float)
    i=0
    for c1 in centroids:
        j=0
        for d2 in dicts:
            point1 = np.asarray(c1['vector'])
            point2 = np.asarray(d2['vector'])
            #Euclidean distance
            distance = np.linalg.norm(point1 - point2)
            matrix[i][j] = distance
            j+=1
        i+=1
    return matrix

#Computes the new centroid for a cluster
def computeMean(traces, events, no):
    mean = {'id':'centroid{}'.format(no)}
    event_vector = []
    for n in range(0, len(events)):
        sum = 0
        if len(traces) != 0:
            for trace in traces:
                sum += trace['vector'][n]
            event_vector.append(sum/len(traces))
        else:
            event_vector.append(0)
    mean['vector']=event_vector
    return mean