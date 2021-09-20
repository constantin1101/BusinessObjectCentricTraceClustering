'''
    Transform a regular event log into a event log that only includes (sampled) variants.
'''


from pm4py.objects.log.obj import EventLog
import pm4py
from pm4py.util import variants_util
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import os

SAMPLED = True

#Method from the pm4py library 
def get_variants_from_log_trace_idx(log, parameters=None):
    """
    Gets a dictionary whose key is the variant and as value there
    is the list of traces indexes that share the variant

    Parameters
    ----------
    log
        Log
    parameters
        Parameters of the algorithm, including:
            Parameters.ACTIVITY_KEY -> Attribute identifying the activity in the log

    Returns
    ----------
    variant
        Dictionary with variant as the key and the list of traces indexes as the value
    """
    if parameters is None:
        parameters = {}

    variants = {}
    for trace_idx, trace in enumerate(log):
        variant = variants_util.get_variant_from_trace(trace, parameters=parameters)
        if variant not in variants:
            variants[variant] = []
        variants[variant].append(trace_idx)

    return variants

log = pm4py.read_xes(os.path.expanduser("~/Downloads/BPI_Challenge_2018.xes"))
variants_trace_idx = get_variants_from_log_trace_idx(log, parameters=None)
print(len(variants_trace_idx))
variants: EventLog
variants = []
if SAMPLED:
    i = 0
    for key in variants_trace_idx:
        if i % 300 == 0:
            for value in variants_trace_idx[key]:
                print(value)
                variants.append(log[value])
                break
        i += 1
else:
    for key in variants_trace_idx:
        for value in variants_trace_idx[key]:
            print(value)
            variants.append(log[value])
            break

print(len(variants))
log._list.clear()
log._list = variants
xes_exporter.apply(log, 'logs/variants/BPI_Challenge_2018_greatly_sampled_variants.xes')