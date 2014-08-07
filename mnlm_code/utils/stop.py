# Model selection / validation criteria tools for LBL and MLBL models

# Stopping criteria:
#  -- maxepoch:         train until the max number of epochs has been reached
#  -- validation_pp:    train until validation log-likelihood no longer improves for k epochs
#  -- ll_train_heldout: train until the (former) heldout set reaches a specified negative logprob


import numpy as np
import gnumpy as gpu
import cPickle as pickle


def save_model(net, loc):
    """
    Save the network model to the specified directory
    """
    output = open(loc, 'wb')
    pickle.dump(net, output)
    output.close()


def load_model(loc):
    """
    Load the network model from the specified directory
    """
    inputs = open(loc, 'rb')
    net = pickle.load(inputs)
    inputs.close()
    return net


def display_results(epoch, time, obj, obj_val, targ=None, targ_val=None):
    """
    Display results 
    """
    if targ == None:
        print "epoch %d... nlogprob:  %.4f, %.4f (%.3f sec)" % (epoch, obj, obj_val, time)
    else:
        print "epoch %d... nlogprob:  %.4f, %.4f, pp: %.4f, %.4f, %.4f, %.4f (%.3f sec)" % (epoch, obj, obj_val, C, C_val, targ, targ_val, time)


def update_result(prev, curr, counter):
    """
    Update the result
    prev: best validation result so far
    curr: current validation result
    counter: epochs since best result
    """
    if curr < prev:
        counter = 0
        return (curr, counter)
    else:
        counter = counter + 1
        return (prev, counter)


def criteria_complete(net, epoch=None, curr=None, obj=None, counter=None, k=None, obj_val=None, target=None):
    """
    Check whether the specified stopping criteria has been met
    """
    criteria = net.criteria
    if criteria == 'maxepoch':
        if epoch == net.maxepoch:
            return True
        else:
            return False
    elif criteria == 'validation_pp':
        if counter >= k:
            return True
        else:
            return False
    elif criteria == 'll_train_heldout':
        if obj_val < target:
            return True
        else:
            return False


def display_phase(phase):
    """
    Print a message displaying the current training phase
    """
    print "============================== Training phase %d ==============================" % (phase)
        


