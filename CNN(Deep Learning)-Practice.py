#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras


# In[2]:


fashion_mnist=keras.datasets.fashion_mnist
(x_train_full,y_train_full),(x_test,y_test)=fashion_mnist.load_data()


# In[3]:


class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
            'Sandal','Shirt','Sneaker','Bag','Ankle boot']


# In[10]:


x_train_full=x_train_full.reshape((60000,28,28,1))
x_test=x_test.reshape((10000,28,28,1))


# In[11]:


x_train_n=x_train_full/255
x_test_n=x_test/225


# In[12]:


x_valid,x_train=x_train_n[:5000],x_train_n[5000:]
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
x_test=x_test_n


# In[13]:


np.random.seed(42)
tf.random.set_seed(42)


# In[16]:


model=keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='valid',activation='relu',input_shape=(28,28,1)))
(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))


# In[17]:


model.summary()


# In[18]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[19]:


model_history=model.fit(x_train,y_train,epochs=30,batch_size=64,validation_data=(x_valid,y_valid))


# In[21]:


import pandas as pd

pd=pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[22]:


ev=model.evaluate(x_test_n,y_test)


# In[26]:


x_new=x_test[:3]


# In[32]:


y_pred=model.predict(x_new)
y_pred


# In[33]:


y_test[:3]


# In[34]:


print(plt.imshow(x_test[0].reshape((28,28))))


# In[ ]:


keras.backe


# In[ ]:





# In[ ]:





# In[ ]:




