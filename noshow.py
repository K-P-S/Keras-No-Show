import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#-----------------------------------------------------------------------

# load dataset
dataframe = pandas.read_csv("mysqlfile.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:16]
Y = dataset[:,16]
#-----------------------------------------------------------------------


#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#requires replacing Y with encoded_Y after results

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=100, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)                   
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
