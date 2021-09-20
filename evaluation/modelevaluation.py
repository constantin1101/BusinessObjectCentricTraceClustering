import pm4py.algo.evaluation.replay_fitness.evaluator as replay_fitness_evaluator
import pm4py.algo.evaluation.precision.evaluator as precision_evaluator
import pm4py.algo.evaluation.generalization.evaluator as generalization_evaluator
import pm4py.algo.evaluation.simplicity.evaluator as simplicity_evaluator

def getReplayFitness(log, net, im, fm, align):
    if align:
        fitness = replay_fitness_evaluator.apply(log, net, im , fm, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    else:
        fitness = replay_fitness_evaluator.apply(log, net, im , fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    return fitness

def getPrecision(log, net, im, fm, align):
    if align:
        precision = precision_evaluator.apply(log, net, im , fm, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    else:
        precision = precision_evaluator.apply(log, net, im , fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    return precision

def getGeneralization(log, net, im, fm):
    generalization = generalization_evaluator.apply(log, net, im, fm)
    return generalization

def getSimplictiy(net):
    simplicity = simplicity_evaluator.apply(net)
    return simplicity
