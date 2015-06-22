This page will describe how to evaluate trained models.

## Evaluation on IAPR TC 12 ##

Assuming you have downloaded the IAPR TC 12 data and model files, open up an IPython session in the main directory and type

```
import proc
(z, zt) = proc.process()
```

This will pre-process the data files and store the results on two dictionaries: _z_ and _zt_. These dictionaries contain all the data, image features and additional structures necessary for training and evaluation. The dictionary _z_ stores all training (and validation) data while _zt_ is used for testing. More details about the contents of these dictionaries can be found [here](AddDetails.md).

Now load a trained model file:

```
from utils import stop
net = stop.load_model('models/mlbl.pkl')
```

The _mlbl_ model corresponds to the additive-biased language model in the paper. The factored three-way model is named _mlblf.pkl_. A log-blinear model (without image conditioning) is also included and referred to as _lbl.pkl_.

To compute perplexity on the test set, run

```
import expr
expr.eval_pp(net, z, zt)

PERPLEXITY: 9.86122706178
```

To compute BLEU scores, run

```
expr.eval_bleu(net, z, zt)

BLEU-1: 0.392735359751
BLEU-2: 0.210988793647
BLEU-3: 0.112200217846
```

Note that the values returned here are better than those reported in the paper. If you perform the same commands with the MLBLF model, you should get output similar to the following:

```
net = stop.load_model('models/mlblf.pkl')
expr.eval_pp(net, z, zt)
expr.eval_bleu(net, z, zt)

PERPLEXITY: 9.89949930326

BLEU-1: 0.387148202494
BLEU-2: 0.209081708733
BLEU-3: 0.115021362301
```

As a baseline, we can also evaluate the LBL model, which does not use any knowledge of images:

```
net = stop.load_model('models/lbl.pkl')
expr.eval_pp(net, z, zt)
expr.eval_bleu(net, z, zt)

PERPLEXITY: 9.29180117955

BLEU-1: 0.321234495687
BLEU-2: 0.145343537401
BLEU-3: 0.0640376403238
```

As expected, the BLEU scores are significantly lower. Full logs from each of the trained models can be found in the _output_ directory.

## Evaluation on other models ##

If you have trained new models (see [here](PageName.md)), they can be evaluated in exactly the same way, simply by loading them using the same command.