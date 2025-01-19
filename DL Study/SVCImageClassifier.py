#TRAINING AN IMAGE CLASSIFIER MODEL

#Importing Libraries
import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metric import accuracy_score
from sklearn.svm import SVC

import pickle

#Preparing Data
input_dir = 'path containing the seperate image folders in local storage'

categories = ['Categ1','Categ2'] #These refer to the two folders in local directory

data = []
labels = []

#Preparing image data from the seperate folders and appending them to the empty data arrays and labels
for category_idx, category in enumerate(categories):
	for file in os.listdir(os.path.join(input_dir, category)):
		img_path = os.path.join(input_dir, category, file)
		img = imread(img_path)
		img = resize(img, (15,15))
		data.append(img.flatten())
		labels.append(category_idx)

#Converting data and labels into numpy arrays		
data = np.asarray(data)
labels = np.asarray(labels)


#Train/test split
x_train, x_test, y_train, y_test = train_test_split(dats, labels
test_size=0.3, shuffle=True, stratify=labels)




#Training Model

#creating instance of SVC
Classifier = SVC()

parameters = [{'gama':[0.01,0.001,0.0001], 'c':[1,10,100,1000]}]

#Training more models(12 models since gama has 3 values and c also has 4 values multiplying to make 12) at ones and assigning them to a variable grid_search
grid_search = GridSearchCV(Classifier, parameters)

grid_search.fit(x_train, y_train)


#Evaluating model

best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)

score = accuracy_score(y_pred, y_test)

print('{}% of samples were correctly clsssified'.format(str(score*100)))

#Save Model

pickle.dump(best_estimstor, open('insert_file_path', 'wb'))

