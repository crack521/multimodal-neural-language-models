This page describes some additional details of the code.

## Description of data dictionaries ##

The output of preprocessing are separate training and testing dictionaries. The training dictionary consists of the following keys:

```
['ngrams',
 'index',
 'text',
 'labels',
 'tokens',
 'instances',
 'word_dict',
 'IM',
 'index_dict',
 'context']
```

Below each of these are described in more detail.

  * ngrams: A list of tuples, each entry corresponding to the n-grams of a caption
  * index: An identifier used to index into the image features
  * text: The raw text loaded from the caption file
  * labels: A large sparse matrix indicating which word (column) appears after which context (row).
  * tokens: The tokenized text
  * instances: Instances used for training the model. Each context window corresponds to an instance.
  * word\_dict: A dictionary that maps vocabulary words to a unique identifier. Each entry from instances is a list of word identifiers.
  * IM: A numpy array of image features
  * index\_dict: The inverse of word\_dict
  * context: A scalar indicating the context size

The dictionary for test examples is of the same format but does not include the word and index dictionaries.

## Description of text pre-processing ##

All of the text is processed using the following operations:

  * All words are mapped to lower case
  * All numerical values are mapped to a special numerical token
  * Special start tokens are appended to the front of captions
  * A special end token is appended to the end of a caption
  * Unseen words (from held-out data) are mapped to a special unknown token
  * Punctuation is treated as words