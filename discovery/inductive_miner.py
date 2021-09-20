import os
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petrinet import visualizer as pn_visualizer

log = pm4py.read_xes("input/logs/HB_extract.xes")
net, initial_marking, final_marking = inductive_miner.apply(log)

tree = inductive_miner.apply_tree(log)

gviz = pt_visualizer.apply(tree)
#pt_visualizer.view(gviz)

net, initial_marking, final_marking = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)