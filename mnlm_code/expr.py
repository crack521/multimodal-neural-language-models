# Model evaluation, generation and retrieval

import numpy as np
from utils import lm_tools
from numpy.random import RandomState
from PIL import Image


def vis(group, indexlist, save=False):
    """
    Visualize images from IAPR TC-12
    """
    ##### Modify these: #####
    imageloc = '/ais/gobi3/u/rkiros/iaprtc12/images/'
    trainloc = 'iaprtc12/iaprtc12_train_list.txt'
    testloc = 'iaprtc12/iaprtc12_test_list.txt'
    #########################

    if group == 'train':
        listloc = trainloc
    else:
        listloc = testloc
    f = open(listloc, 'rb')
    ims = []
    for line in f:
        ims.append(line.strip() + '.jpg')
    f.close()
    for i in range(len(indexlist)):
        imloc = imageloc + ims[indexlist[i]]
        im = Image.open(imloc)
        im.thumbnail((256,256), Image.ANTIALIAS)
        im.show()
        if save:
            im.save('r' + str(i) + '.jpg')


def eval_pp(net, z, zt):
    """
    Evaluate the perplexity of net
    z: training dictionary
    zt: testing dictionary
    """
    if net.name != 'lbl':
        Im = zt['IM']
    else:
        Im = None
    pp = lm_tools.perplexity(net, zt['ngrams'], z['word_dict'], Im=Im, context=net.context)
    print 'PERPLEXITY: ' + str(pp)


def eval_bleu(net, z, zt):
    """
    Evaluate BLEU scores of samples from net
    z: training dictionary
    zt: testing dictionary
    """
    if net.name != 'lbl':
        Im = zt['IM']
    else:
        Im = None
    bleu = lm_tools.compute_bleu(net, z['word_dict'], z['index_dict'], zt['tokens'], IM=Im)
    bleu_means = np.mean(bleu, 0)
    print 'BLEU-1: ' + str(bleu_means[0])
    print 'BLEU-2: ' + str(bleu_means[1])
    print 'BLEU-3: ' + str(bleu_means[2])
    
    
def generate(net, z, maxlen=50, im=None, init=None, use_end=True):
    """
    Generate a sample from the model net
    """
    caption = lm_tools.sample(net, z['word_dict'], z['index_dict'], num=maxlen, Im=im, initial=init, use_end=use_end)
    print ' '.join(caption)


def im2txt(net, z, im, k=5, shortlist=15):
    """
    Given image query im, retrieve the top-k captions from tokens
    """
    captions = lm_tools.im2txt(net, im, z['word_dict'], z['tokens'], z['IM'], k=k, shortlist=shortlist)
    for c in captions:
        print ' '.join(c)
    

def txt2im(net, z, txt, k=5, search=100, seed=1234):
    """
    Given text query txt, retrieve the top-k images from z['IM']
    For speed, only searches over a random subset of 'search' images
    """
    inds = np.arange(len(z['IM']))
    prng = RandomState(seed)
    prng.shuffle(inds)  
    ims = lm_tools.txt2im(net, txt, z['IM'][inds[:search]], z['word_dict'], k=k)
    return inds[ims]




