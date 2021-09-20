import numpy as np

#Transforms traces into 'business-object' lists and 'actions' lists
def createDicts(event_log):
    object_dicts = []
    action_dicts = []

    i = 1
    for id in event_log.case_id.unique() :
        dict_trace_objects = {"id": i}
        dict_trace_actions = {"id": i}
        trace = event_log[event_log["case_id"]==id]
        object_list = []
        action_list = []
        for object in trace["business_object"]:
            object_list.append(object)
        for action in trace["action"]:
            action_list.append(action)
        dict_trace_objects['list']=object_list
        dict_trace_actions['list']=action_list
        object_dicts.append(dict_trace_objects)
        action_dicts.append(dict_trace_actions)
        i += 1

    return object_dicts, action_dicts

#Augmented Levensthein distance based on the dynamic programming approach
def levenstheinDistanceAugmented(first_BO, second_BO, first_A, second_A):
    matrix = np.zeros((len(first_BO)+1,len(second_BO)+1))

    for i in range(len(first_BO)+1):
        matrix[i, 0] = i
    
    for j in range(len(second_BO)+1):
        matrix[0, j] = j
    
    for i in range(len(first_BO)+1): 
        for j in range(len(second_BO)+1): 
            if first_BO[i-1] == second_BO[j-1]:
                if first_A[i-1] != second_A[j-1]:
                    substitutionCost = 0.5
                else:
                    substitutionCost = 0
            else:
                substitutionCost = 1
            
            matrix[i,j] = min(matrix[i][j-1] + 1,  
                                matrix[i-1][j] + 1,        
                                matrix[i-1][j-1] + substitutionCost)

    return matrix[len(first_BO)][len(second_BO)]

#Coputes the new centroid for a cluster
def computeCentroid(object_traces, action_traces, no):
    object_centroid = {'id':'centroid{}'.format(no)}
    action_centroid = {'id':'centroid{}'.format(no)}
    matrix = computeDistanceMatrix(object_traces, action_traces, object_traces, action_traces)
    min = 100
    id = 0

    for i in range(0, len(object_traces)):
        sum = 0
        for j in range(0, len(object_traces)):
            sum += matrix[i][j]
        average = sum/len(object_traces)
        if average < min:
            min = average
            id = i

    object_centroid['list'] = object_traces[id]['list']
    action_centroid['list'] = action_traces[id]['list']

    return object_centroid, action_centroid

#Computes the distance matrix between centroids and traces
def computeDistanceMatrix(object_centroids, action_centroids, object_dicts, action_dicts):
    matrix = np.zeros((len(object_centroids),len(object_dicts)), dtype=np.float)

    i=0
    for oc, ac in zip(object_centroids, action_centroids):
        j=0
        for od, ad in zip(object_dicts, action_dicts):
            first_BO = np.asarray(oc['list'])
            second_BO = np.asarray(od['list'])
            first_A = np.asarray(ac['list'])
            second_A = np.asarray(ad['list'])
            distance = levenstheinDistanceAugmented(first_BO, second_BO, first_A, second_A)
            matrix[i][j] = distance
            j+=1
        i+=1

    return matrix