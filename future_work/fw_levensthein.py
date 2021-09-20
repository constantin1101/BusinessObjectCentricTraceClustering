import numpy as np

def futureWork_createDicts(event_log):
    object_dicts = []
    action_dicts = []
    actor_dicts = []

    i = 1
    for id in event_log.case_id.unique() :
        dict_trace_objects = {"id": i}
        dict_trace_actions = {"id": i}
        dict_trace_actor = {"id": i}
        trace = event_log[event_log["case_id"]==id]
        object_list = []
        action_list = []
        actor_list = []
        for object in trace["business_object"]:
            object_list.append(object)
        for action in trace["action"]:
            action_list.append(action)
        for actor in trace["actor"]:
            actor_list.append(actor)
        dict_trace_objects['list']=object_list
        dict_trace_actions['list']=action_list
        dict_trace_actor['list']=actor_list
        object_dicts.append(dict_trace_objects)
        action_dicts.append(dict_trace_actions)
        actor_dicts.append(dict_trace_actor)
        i += 1

    return object_dicts, action_dicts, actor_dicts

def futureWorkDistance(first_BO, second_BO, first_A, second_A, first_R, second_R):
    matrix = np.zeros((len(first_BO)+1,len(second_BO)+1))

    for i in range(len(first_BO)+1):
        matrix[i, 0] = i
    
    for j in range(len(second_BO)+1):
        matrix[0, j] = j
    
    for i in range(len(first_BO)+1): 
        for j in range(len(second_BO)+1): 
            if first_BO[i-1] == second_BO[j-1]:
                if first_A[i-1] != second_A[j-1]:
                    if first_R[i-1] != second_R[j-1]:
                        substitutionCost = 1
                    else:
                        substitutionCost = 0.5
                else:
                    if first_R[i-1] != second_R[j-1]:
                        substitutionCost = 0.5
                    else:
                        substitutionCost = 0
            else:
                substitutionCost = 1
            
            matrix[i,j] = min(matrix[i][j-1] + 1,  
                                matrix[i-1][j] + 1,        
                                matrix[i-1][j-1] + substitutionCost)

    return matrix[len(first_BO)][len(second_BO)]

def futureWork_computeCentroid(object_traces, action_traces, actor_traces, no):
    object_centroid = {'id':'centroid{}'.format(no)}
    action_centroid = {'id':'centroid{}'.format(no)}
    actor_centroid = {'id':'centroid{}'.format(no)}

    if len(object_traces) > 0:
        matrix = futureWork_computeDistanceMatrix(object_traces, action_traces, actor_traces, object_traces, action_traces, actor_traces)
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
        actor_centroid['list'] = actor_traces[id]['list']

        return object_centroid, action_centroid, action_centroid

    else:
        return 0, 0, 0

def futureWork_computeDistanceMatrix(object_centroids, action_centroids, actor_centroids, object_dicts, action_dicts, actor_dicts):
    matrix = np.zeros((len(object_centroids),len(object_dicts)), dtype=np.float)

    i=0
    for oc, ac, rc in zip(object_centroids, action_centroids, actor_centroids):
        j=0
        for od, ad, rd in zip(object_dicts, action_dicts, actor_dicts):
            first_BO = np.asarray(oc['list'])
            second_BO = np.asarray(od['list'])
            first_A = np.asarray(ac['list'])
            second_A = np.asarray(ad['list'])
            first_R = np.asarray(rc['list'])
            second_R = np.asarray(rd['list'])
            distance = futureWorkDistance(first_BO, second_BO, first_A, second_A, first_R, second_R)
            matrix[i][j] = distance
            j+=1
        i+=1

    return matrix