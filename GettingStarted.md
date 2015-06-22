## Requirements ##

Python 2.7, Numpy, Scipy and NLTK.

  * For GPU use, you should have cudamat installed (https://github.com/cudamat/cudamat) and have its directly on your LD\_LIBRARY path. Included in this package is gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html) and npmat (http://www.cs.utoronto.ca/~ilya/npmat.py). The later will allow you to run the code with CPU usage only.

  * All of our models were trained using image features from the Toronto Convnet (https://github.com/TorontoDeepLearning/convnet). This is required to install if you want to use our trained models with new images.

All code was tested on 64-bit Ubuntu with GTX 5xx GPUs.

## Checking out the code ##

You can check out the code from trunk with this line:

```
svn checkout http://multimodal-neural-language-models.googlecode.com/svn/trunk/ multimodal-neural-language-models-read-only
```

## Downloads ##

Download the following files from http://www.cs.toronto.edu/~rkiros/multimodal.html and unpack these in the main directory:

  * IAPR TC-12 data and image features
  * IAPR TC-12 trained models

Our models are pre-trained using the publicly available 50-dimensional scaled representations from Turian et al. These are recommended for training new models. Download them from here: http://metaoptimize.com/projects/wordreprs/ and place them in the 'embeddings' directory.
