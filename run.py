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
# Data Processing
import  data_pipeline
# Evaluation
from evaluation import EvaluationClass
# Genetic Algorithm
from GA import GA


"""**************Hyperparameter Specification**************"""
Q = 0.05 # Tail Quantile
F = 3 # Length of Forword Window
train_start = "20120808" # The Starting Point of Training Dataset
train_sample_num = 30 # The number of training ( factor, performance) pairs
test_start = "20130204" # The Starting Point of Testing Dataset
test_sample_num = 5 # The number of training ( factor, performance) pairs
    
    
"""**************Loading Data**************"""
print("- Loading Data")
import pickle as pk
with open('data.pkl','rb') as f:
    data = pk.load(f)
X_train, Y_train = data['X_train'], data['Y_train']
#data_pipeline.load( train_start, train_sample_num, F, Q)
X_test, Y_test = data['X_test'], data['Y_test']
#data_pipeline.load( test_start, test_sample_num, F, Q) 
database = { "X_train":X_train, "Y_train":Y_train, "X_test":X_test, "Y_test":Y_test}

    
"""**************Logistic Regression based Stock Selection**************"""
print("- Logistic Regression based Stock Selection")
model_logistic = EvaluationClass( X_train, Y_train, X_test, Y_test, model_type = 1)
LAccuracy, LPrecision, LRecall, LF1, LTPR, LFPR, LAUC = model_logistic.evalu()

        
"""**************Deep Learning based Stock Selection**************"""
print("- Deep Learning based Stock Selection")
model_deep = EvaluationClass( X_train, Y_train, X_test, Y_test, model_type = 2)
DAccuracy, DPrecision, DRecall, DF1, DTPR, DFPR, DAUC = model_deep.evalu()
    
    
"""**************Random Forest based Stock Selection**************"""   
print("- Random Forest based Stock Selection")
model_forest = EvaluationClass( X_train, Y_train, X_test, Y_test, model_type = 3)
RAccuracy, RPrecision, RRecall, RF1, RTPR, RFPR, RAUC = model_forest.evalu()


"""Stacking based Stock Selection"""
print("- Stacking based Stock Selection")
model_stacking = EvaluationClass( X_train, Y_train, X_test, Y_test, model_type = 4)
SAccuracy, SPrecision, SRecall, SF1, STPR, SFPR, SAUC = model_stacking.evalu()


"""**************Genetic Algorithm based Feature Selection**************""" 
print("- GA based Feature Selection")
Selector = GA( X_train, Y_train, X_test, Y_test, model_type = 1, save_computaion=True)
Selector.Search()
Filter = Selector.bestSolutions
X_train_masked = X_train[ :, Filter==1]
X_test_masked = X_test[ :, Filter==1]


"""**************Logistic Regression based Stock Selection (After Feature Selection)**************"""
print("- Logistic Regression based Stock Selection (After Feature Selection)")
model_logistic_masked = EvaluationClass( X_train_masked, Y_train, X_test_masked, Y_test, model_type = 1)
LAccuracy_masked, LPrecision_masked, LRecall_masked, LF1_masked, LTPR_masked, LFPR_masked, LAUC_masked = model_logistic_masked.evalu()

        
"""**************Deep Learning based Stock Selection (After Feature Selection)**************"""
print("- Deep Learning based Stock Selection (After Feature Selection)")
model_deep_masked = EvaluationClass( X_train_masked, Y_train, X_test_masked, Y_test, model_type = 2)
DAccuracy_masked, DPrecision_masked, DRecall_masked, DF1_masked, DTPR_masked, DFPR_masked, DAUC_masked = model_deep_masked.evalu()
    
    
"""**************Random Forest based Stock Selection (After Feature Selection)**************"""   
print("- Random Forest based Stock Selection (After Feature Selection)")
model_forest_masked = EvaluationClass( X_train_masked, Y_train, X_test_masked, Y_test, model_type = 3)
RAccuracy_masked, RPrecision_masked, RRecall_masked, RF1_masked, RTPR_masked, RFPR_masked, RAUC_masked = model_forest_masked.evalu()


"""Stacking based Stock Selection"""
print("- Stacking based Stock Selection (After Feature Selection)")
model_stacking_masked = EvaluationClass( X_train_masked, Y_train, X_test_masked, Y_test, model_type = 4)
SAccuracy_masked, SPrecision_masked, SRecall_masked, SF1_masked, STPR_masked, SFPR_masked, SAUC_masked = model_stacking_masked.evalu()
