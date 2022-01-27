import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications.inception_v3 import InceptionV3

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


### linux ###
# !unzip -q -o /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip
# !unzip -q -o /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip

file_list = glob.glob('train/*')
file_list[ : 10]

len(file_list)

Image.open('train/dog.3666.jpg')

train = pd.DataFrame({'paht' : file_list})
train['label'] = train['path'].apply(lambda x : x.split('/')[-1].split('.')[0])
print(train)

idg = ImageDataGenerator()
train_generator = idg.flow_from_dataframe(train, x_col = 'path', y_col = 'label', target_size = (100, 100), batch_size = 128)

iv = InceptionV3(include_top  = False, pooling = 'avg')

model = Sequential()
model.add(iv)
model.add(Dense(2, activation = 'softmax'))
model.compile(metrics = 'acc', optimizer = 'adam', loss = 'categorical_crossentropy')
model.fit(train_generator)

test = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition-sample_submission.csv')
print(test)

glob.glob('test/*')[ : 10]

print(test.dtypes)

test['path'] = 'test/' + test['id'].astype('str') + '.jpg'
print(test[ : 10])

test_generator = idg.flow_to_dataframe(test, x_col = 'path', y_col = None, target_size = (100, 100), batch_size = 128, class_mode = None, shuffle = False)
result = model.predict(test_generator, verbose = 1, workers = 2)

print(result)