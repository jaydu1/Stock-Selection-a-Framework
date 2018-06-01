"""
Stock Selection ----- Deep or Traditional Methods?
    
Authors:
XingYu Fu; JinHong Du; YiFeng Guo; MingWen Liu; Tao Dong; XiuWen Duan; 
ZiYi Yang; MingDi Zheng; Yuan Zeng;

Institutions:
AI&Fintech Lab of Likelihood Technology; 
Gradient Trading;
School of Mathematics, Sun Yat-sen University;
LingNan College, Sun Yat-sen University;

Contact:
fuxy28@mail2.sysu.edu.cn

All Rights Reserved.
"""


"""Import Modules"""
# Numerical Computation
import numpy as np
from sklearn.metrics import roc_auc_score
# Plot
#import matplotlib.pyplot as plt
# Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
# Random Forest 
from sklearn.ensemble import RandomForestRegressor

class EvaluationClass:
    def __init__( self, X_train, Y_train, X_test, Y_test, model_type, save_computaion=False):
        
        if model_type == 1: # Logistic Regression
            input_shape = np.shape(X_train)[1]
            model = Sequential()
            model.add(Dense(1, input_dim=input_shape, activation='sigmoid'))
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model = model
            self.Y_train = Y_train
            self.Y_test = Y_test
            self.X_train = X_train
            self.X_test = X_test
            
        elif model_type == 2: # Deep Learning Model
            input_shape = np.shape(X_train)[1]
            model = Sequential()
            model.add(Dense(input_shape//2, activation='relu', input_dim=input_shape))
            model.add(Dropout(0.5))
            model.add(Dense(input_shape//4, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(2, activation='softmax'))
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) 
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model = model
            self.Y_train = to_categorical(Y_train, num_classes=2)
            self.Y_test = to_categorical(Y_test, num_classes=2)
            self.X_train = X_train
            self.X_test = X_test

        elif model_type == 3: # Random Forest
            model = RandomForestRegressor(n_estimators = 100, max_depth = 4)
            self.model = model
            self.Y_train = Y_train
            self.Y_test = Y_test  
            self.X_train = X_train
            self.X_test = X_test
            
        else : # Stacking
            # NN
            input_shape = np.shape(X_train)[1]
            train_sample_num = np.shape(X_train)[0]
            model1 = Sequential()
            model1.add(Dense(input_shape//2, activation='relu', input_dim=input_shape))
            model1.add(Dropout(0.5))
            model1.add(Dense(input_shape//4, activation='relu'))
            model1.add(BatchNormalization())
            model1.add(Dense(2, activation='softmax'))
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) 
            model1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model1 = model1
            self.X_train1 = X_train[:-train_sample_num//5]
            self.Y_train1 = to_categorical(Y_train, num_classes=2)[:-train_sample_num//5]
            # RF
            model2 = RandomForestRegressor(n_estimators = 100, max_depth = 4)
            self.model2 = model2
            self.X_train2 = X_train[:-train_sample_num//5]
            self.Y_train2 = Y_train[:-train_sample_num//5]
            # Logstic
            model3 = Sequential()
            model3.add(Dense(1, input_dim=2, activation='sigmoid'))
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model3.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model3 = model3
            self.X_train3 = X_train[-train_sample_num//5:]
            self.Y_train3 = Y_train[-train_sample_num//5:]
            self.X_test = X_test
            self.Y_test = Y_test

        self.model_type = model_type
        self.save_computaion = save_computaion
    
    
    """Definition of the Back Test System"""
    def evalu(self):
    
        """*********************Training*********************"""    
        if (self.model_type == 1) or (self.model_type == 2):
            if self.save_computaion: # For GA Feature Selection
                self.model.fit(self.X_train, self.Y_train, epochs=5, batch_size=128, verbose=0)
            else:
                self.model.fit(self.X_train, self.Y_train, epochs=20, batch_size=128, verbose=1)
        elif self.model_type == 3:
            self.model.fit(self.X_train, self.Y_train)
        else:
            self.model1.fit(self.X_train1, self.Y_train1, epochs=20, batch_size=128, verbose=0)
            self.model2.fit(self.X_train2, self.Y_train2)
            Y1 = self.model1.predict( self.X_train3 )[:,1]
            Y2 = self.model2.predict( self.X_train3 )
            X3 = np.array( [ [ Y1[i], Y2[i]] for i in range(len(Y1))] )
            self.model3.fit(X3, self.Y_train3, epochs=30, batch_size=128, verbose=1)
    
    
        """*********************Prediction*********************"""
        if self.model_type == 1 or self.model_type == 3:
            Y_continuous = self.model.predict( self.X_test )
            Y_test = self.Y_test
        elif self.model_type == 2:
            Y_continuous = self.model.predict( self.X_test )[:,1]
            Y_test = self.Y_test[:,1]
        else:
            Y_continuous1 = self.model1.predict( self.X_test )[:,1]
            Y_continuous2 = self.model2.predict( self.X_test )
            X3 = np.array( [ [ Y_continuous1[i], Y_continuous2[i]] for i in range(len(Y_continuous1))] )
            Y_continuous = self.model3.predict( X3 )
            Y_test = self.Y_test
        Y_discrete = np.round( Y_continuous )
    
        """*********************Statistical Evaluating*********************"""
        if self.save_computaion:
           # AUC
            AUC = roc_auc_score(Y_test, Y_continuous)
            return AUC 
        else:
            # TP; FP; FN; TN
            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range( len(Y_discrete) ):
                if Y_discrete[i] == Y_test[i]:
                    if Y_discrete[i] == 1: 
                        TP += 1
                    else:
                        TN += 1
                else:
                    if Y_discrete[i] == 1:
                        FP += 1
                    else:
                        FN += 1
            
            # Accuracy
            Accuracy = (TP+TN)/(TP+TN+FP+FN)
            # Precision
            Precision = TP/(TP+FP) 
            #Recall
            Recall = TP/(TP+FN)
            # F1-score
            F1 = 2*Precision*Recall/(Precision+Recall)
            # TPR
            TPR = TP/(TP+FN)
            # FPR
            FPR = FP/(FP+TN)
            # AUC
            AUC = roc_auc_score(Y_test, Y_continuous)
            return Accuracy, Precision, Recall, F1, TPR, FPR, AUC
