import os
import pm4py
import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

def getClusterLog(labels, eventlog_df, clusterNo):
    for i in range(0, len(labels)):
        if labels[i] != clusterNo:
            eventlog_df = eventlog_df[eventlog_df['case_id']!=(i+1)]
    eventlog_df = eventlog_df.rename(columns={'case_id':'case:concept:name', 'event':'concept:name'})
    return pm4py.convert_to_event_log(eventlog_df)

def alphaMiner(log):
    net, initial_marking, final_marking = alpha_miner.apply(log)
    return net, initial_marking, final_marking

def heuristicMiner(log):
    net, initial_marking, final_marking = heuristics_miner.apply(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
    return net, initial_marking, final_marking

def inductiveMiner(log):
    net, initial_marking, final_marking = inductive_miner.apply(log)
    return net, initial_marking, final_marking