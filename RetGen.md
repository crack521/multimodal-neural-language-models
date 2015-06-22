This page describes how to perform retrieval and generation with trained models, using the IAPR TC-12 dataset as an example.

## Preliminaries ##

In order to view results, you need to download the images from IAPR TC-12. You can download the dataset from here: http://imageclef.org/photodata

Once you have unpacked the data, open up the module _expr.py_ in the main directory and edit the following lines from the _vis_ function:

```
imageloc = '/ais/gobi3/u/rkiros/iaprtc12/images/'
trainloc = 'iaprtc12/iaprtc12_train_list.txt'
testloc = 'iaprtc12/iaprtc12_test_list.txt'
```

Set _imageloc_ to be the location of the IAPR TC-12 images. The later two lines should not need changing unless you are using a different dataset. Once this is complete, the _vis_ function can be used for visualizing retrieved images.

## Retrieving images from a text query ##

You can retrieve training images from a new query by running the following:

```
images = expr.txt2im(net, z, 'tennis match'.split(), k=3, search=1000)
```

Given the query 'tennis match', the output will be a list of indices from the training set, corresponding to what the model thinks are the best images for the query. You can visualize these images by running

```
expr.vis('train', images)
```

With the _mlbl_ model, this returns the following images:

![http://www.cs.toronto.edu/~rkiros/images/result_0.jpg](http://www.cs.toronto.edu/~rkiros/images/result_0.jpg) ![http://www.cs.toronto.edu/~rkiros/images/result_1.jpg](http://www.cs.toronto.edu/~rkiros/images/result_1.jpg) ![http://www.cs.toronto.edu/~rkiros/images/result_2.jpg](http://www.cs.toronto.edu/~rkiros/images/result_2.jpg)

The _search_ argument indicates how many randomly selected training images to search for. As another example:

```
images = expr.txt2im(net, z, 'sunset'.split(), k=3, search=2500)
expr.vis('train', images)
```

Which outputs the following images:

![http://www.cs.toronto.edu/~rkiros/images/r0.jpg](http://www.cs.toronto.edu/~rkiros/images/r0.jpg)
![http://www.cs.toronto.edu/~rkiros/images/r1.jpg](http://www.cs.toronto.edu/~rkiros/images/r1.jpg)
![http://www.cs.toronto.edu/~rkiros/images/r2.jpg](http://www.cs.toronto.edu/~rkiros/images/r2.jpg)

If you are not satisfied with the results, try increasing the _search_ argument. This is slower but gives a larger space to search over.

## Retrieving text from an image query ##

To retrieve text from an image query, enter

```
expr.im2txt(net, z, zt['IM'][1], k=3, shortlist=15)
expr.vis('test', [1])
```

This uses the test image at index 1, and returns the following three captions from the training set:

![http://www.cs.toronto.edu/~rkiros/images/f0.jpg](http://www.cs.toronto.edu/~rkiros/images/f0.jpg)
```
thunderous waterfall over a high cliff ;
a waterfall over high dark grey rocks with a few green bushes in the background ;
a small waterfall over a black and brown cliff surrounded by green bushes and brown grass ;
```

The _shortlist_ argument selects how many images that are nearest to the query image to use in order to compute perplexities. Another example:

```
expr.im2txt(net, z, zt['IM'][11], k=3, shortlist=25)
expr.vis('test', [11])
```

This uses the test image at index 11, and returns the following three captions from the training set:

![http://www.cs.toronto.edu/~rkiros/images/f1.jpg](http://www.cs.toronto.edu/~rkiros/images/f1.jpg)
```
two tourists are standing in a classroom with many tourists at their wooden desks ;
a group of nine people is standing in a classroom with yellow and white walls and a dark green board ;
tourists are standing in a classroom ; local pupils are sitting at their wooden desks ;

```

## Generating text conditioned on images ##

To generate text, use the following command:

```
expr.generate(net, z, im=zt['IM'][15])
expr.vis('test', [15])
```

Conditioned on the test image at index 15, the model generates the following:

![http://www.cs.toronto.edu/~rkiros/images/g0.jpg](http://www.cs.toronto.edu/~rkiros/images/g0.jpg)

```
a dark - skinned , dark - skinned girl wearing a white pullover shirts is wearing a red flower ; <end>
```

By default, the caption generator runs until it generates an _end_ token, in which case it stops. Another example:

```
expr.generate(net, z, im=zt['IM'][100])
expr.vis('test', [100])
```

Conditioned on the test image at index 100, the model generates the following:

![http://www.cs.toronto.edu/~rkiros/images/g1.jpg](http://www.cs.toronto.edu/~rkiros/images/g1.jpg)
```
two people and a statue of a city at night ; <end>
```

In practice it might be beneficial to generate several captions and return the ones which give the lowest perplexity given the image (although this is never done in our paper).

## Using new images ##

To apply our models to new images (that don't appear in IAPR TC-12), you need to install Convnet (https://github.com/TorontoDeepLearning/convnet) and apply the feature extractor to your images e.g.

```
extract_representation_cpu --model CLS_net_20140621074703.pbtxt --parameters CLS_net_20140621074703.h5 --mean pixel_mean.h5 --output cpu_out --layer hidden7 --layer output  < images.txt
```

where images.txt are the locations of your images. After loading the text outputs into a numpy array, these can be fed into any of the above functions.