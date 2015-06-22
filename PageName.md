This page describes how to train new models.

## Pre-processing the data ##

In the main directory, open the module _proc.py_ and edit the following lines:

```
train_captions = 'iaprtc12/train_captions.txt'
train_images = 'iaprtc12/train_hidden7.txt'
test_captions = 'iaprtc12/test_captions.txt'
test_images = 'iaprtc12/test_hidden7.txt'
context = 5
```

If you want to train a new model on IAPR TC-12, then the first 4 lines don't require any modification. If you want to use a new dataset, specify the locations of the training and test captions, as well as the training and test image features. Each caption should appear on a single line. This is the first few lines of train\_captions.txt:

```
a yellow building with white columns in the background ; two palm trees in front of the house ; cars parked in front of the house ; a woman and a child are walking over the square ;
a white , large statue with spread arms on a hill ; picture taken from behind ; bushes and small trees on the hill ; street leading up the hill ; there are white clouds in the blue sky ;
view of a city with many dark green trees and white houses with red roofs ; there are a few high - rise buildings and a stadium in the centre ;
```

Consequently, each of the image features should be in the same order as the captions. By default, the image features are stored in a txt file for simplicity. In case this becomes prohibitive, edit the following function in _proc.py_ to modify how loading is done:

```
def load_convfeatures(loc):
    """
    Reads in the txt file produces by ConvNet
    Consider modifying this for other file types (e.g .npy)
    """
    return np.loadtxt(loc)
```

Once you are happy, save _proc.py_, reload it and enter the following:

```
(z, zt) = proc.process()
```

To train new models, each type of model has its own trainer function. Suppose that we would like to train a new MLBLF model. First import the trainer:

```
from trainers import trainer_mlblf
```

If you open trainer\_mlblf, you can modify all the options and hyperparameters for training the models:

```
net = mlblf.MLBLF(name='mlblf',
                      loc='models/mlblf.pkl',
                      seed=1234,
                      criteria='validation_pp',
                      k=5,
                      V=vocabsize,
                      K=50,
                      D=im.shape[1],
                      h=256,
                      factors=50,
                      context=context,
                      batchsize=20,
                      maxepoch=100,
                      eta_t=0.02,
                      gamma_r=1e-4,
                      gamma_c=1e-5,
                      f=0.998,
                      p_i=0.5,
                      p_f=0.9,
                      T=20.0,
                      verbose=1)
```

The above values are those used for training the models included with the package. Below each of these are described:

  * loc: where to store the model file
  * seed: the seed used for initializing the model parameters
  * criteria: the stopping criteria. validation\_pp uses the log-likelihood on the validation set to control training. Alternatively, using maxepoch will run the net for the max number of epochs specified below, saving the net as the validation results improve.
  * k: the window size used for validation
  * V: the size of the vocabulary
  * K: the dimensionality of the word representations
  * D: the dimensionality of the images features (4096 for convnet features, MLBL and MLBLF only)
  * h: dimensionality of an intermediate layer on the image channel (MLBL and MLBLF only)
  * factors: number of factors (MLBLF only)
  * context: the context size
  * batchsize: size of the minibatches used for SGD
  * maxepoch: max number of epochs to run before training is stopped
  * eta\_t: the initial learning rate
  * gamma\_r: weight decay on the word representations
  * gamma\_c: weight decay on the context matricies
  * f: multiplicative per-epoch learning rate decay
  * p\_i: the initial momentum
  * p\_f: the final momentum
  * T: The number of epochs to go from initial to final momentum (linearly)
  * verbose: whether to print anything

Once you are satisfied, save the changes, go to the main directory and run

```
trainer_mlblf.trainer(z, split=3500, pre_train=True)
```

This will launch the training. Here, split indicates the number of validation examples, while pretrain indicates whether or not to use pre-trained word embeddings. If you haven't downloaded any, set this to False.

As training is being done, per-epoch output will be printed. After 10 epochs of training the MLBLF model, you should get a result similar to this:

```
============================== Training phase 1 ==============================
epoch 0... nlogprob:  2.7754, 2.8075 (532.314 sec)
epoch 1... nlogprob:  2.5209, 2.5955 (513.821 sec)
epoch 2... nlogprob:  2.3709, 2.4715 (525.327 sec)
epoch 3... nlogprob:  2.2648, 2.3987 (535.004 sec)
epoch 4... nlogprob:  2.1830, 2.3414 (524.772 sec)
epoch 5... nlogprob:  2.1230, 2.3194 (535.810 sec)
epoch 6... nlogprob:  2.0781, 2.3017 (533.634 sec)
epoch 7... nlogprob:  2.0121, 2.2601 (536.419 sec)
epoch 8... nlogprob:  1.9685, 2.2351 (524.698 sec)
epoch 9... nlogprob:  1.9268, 2.2338 (535.200 sec)
```

The columns of numbers correspond to the negative log probabilities on the training and validation sets, respectively. In brackets is the number of seconds it takes to complete an epoch. From this example, each epoch takes just under 9 minutes.

To load a new model, run the following:

```
net = stop.load_model('models/mlbl.pkl')
```

where the argument is the model destination (i.e. the _loc_ argument). Once the model is loaded, it can be passed to any of the _expr_ functions (for evaluation, retrieval, generation, etc)