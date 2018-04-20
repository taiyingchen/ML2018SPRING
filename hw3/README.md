# Machine Learning Homework 3 - Image Sentiment Classification

## Description

本次作業為Image Sentiment Classification。我們提供給各位的training dataset為兩萬八千張左右48x48 pixel的圖片，以及每一張圖片的表情label（注意：每張圖片都會唯一屬於一種表情）。總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。 Testing data則是七千張左右48x48的圖片，希望各位同學能利用training dataset訓練一個CNN model，預測出每張圖片的表情label（同樣地，為0~6中的某一個）並存在csv檔中。

[注意] 這次作業希望大家在衝高Kaggle上Accuracy的同時，對訓練的model及預測的結果多做一些觀察（P3-P5），並在報告中多加詳述。

## Training Data Description

```python
df.head()
df.describe()
df['column'].value_counts()
```

label   counts
3       7215
6       4965
4       4830
2       4097
0       3995
5       3171
1        436

## Procedure

原始label為厭惡的training data太少，僅有436筆，導致predict不出厭惡的label -> 以左右鏡相翻轉再得到436筆training data

append new training data造成training set後段全為new training data -> shuffle images and labels

## Tips

### One Hot Encoding & Decoding

np.to_categorical <-> np.argmax

### Load Keras Model

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

## Reference

[[Python] Pandas 基礎教學](https://oranwind.org/python-pandas-ji-chu-jiao-xue/)
[學習使用Keras建立卷積神經網路](https://chtseng.wordpress.com/2017/09/23/%E5%AD%B8%E7%BF%92%E4%BD%BF%E7%94%A8keras%E5%BB%BA%E7%AB%8B%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF/)
[Data Augmentation 資料增強](https://chtseng.wordpress.com/2017/11/11/data-augmentation-%E8%B3%87%E6%96%99%E5%A2%9E%E5%BC%B7/)
[How can I save a Keras model?](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)

### Batch Normalization

[Why does Batch Norm work?](https://www.coursera.org/learn/deep-neural-network/lecture/81oTm/why-does-batch-norm-work)

### Confusion Matrix

[scikit-learn toolkit](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)