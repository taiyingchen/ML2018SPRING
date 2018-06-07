# Machine Learning Homework 5

## Requirements

Toolkit version

* tensorflow1.4.0
* keras2.0.8
* pytorch0.3.0
* gensim3.1.0
* GloVe0.1.0

## Description

labeled training data：20萬
unlabeled training data：120萬
testing data：20萬（10萬public，10萬private）

## Tips

### `keras.preprocessing.text.Tokenizer`

This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...

`0` is a reserved index that won't be assigned to any word.

## References

### string preprocessing

[How to remove punctuation marks from a string in Python 3.x using .translate()?](https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)

[Python: Best Way to remove duplicate character from string](https://stackoverflow.com/questions/18799036/python-best-way-to-remove-duplicate-character-from-string?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)

### gensim

[以 gensim 訓練中文詞向量](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)

[From Strings to Vectors](https://radimrehurek.com/gensim/tut1.html#from-strings-to-vectors)

[models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

[python 下的 word2vec 学习笔记](https://blog.csdn.net/jerr__y/article/details/52967351)

### RNN

[Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

[Keras 模型中使用预训练的 gensim 词向量和可视化](https://eliyar.biz/using-pre-trained-gensim-word2vector-in-a-keras-model-and-visualizing/)

[Guide To Word Vectors With Gensim And Keras](https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/)

[keras：3)Embedding层详解](https://blog.csdn.net/jiangpeng59/article/details/77533309)

[Ensemble and Store Models in Keras 2.x](https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f)