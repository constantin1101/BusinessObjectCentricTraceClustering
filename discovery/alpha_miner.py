import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

log = pm4py.read_xes("input/logs/HB_extract.xes")
net, initial_marking, final_marking = alpha_miner.apply(log)