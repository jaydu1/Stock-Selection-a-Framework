"""
Stock Selection ----- Deep or Traditional Methods?
    
Authors:
XingYu Fu; JinHong Du; YiFeng Guo; MingWen Liu; Tao Dong; XiuWen Duan; 
ZiYi Yang; MingDi Zheng; Yuan Zeng;

Institutions:
AI&Fintech Lab of Gradient Trading; 
School of Mathematics, Sun Yat-sen University;
LingNan College, Sun Yat-sen University;

Contact:
fuxy28@mail2.sysu.edu.cn

All Rights Reserved.
"""


"""Import Modules"""
# Numerical Computation
import numpy as np
from math import sqrt
# Load Data
import pandas as pd
import os


"""Load Factor-Data from Disk"""
def load( start, sample_num, F, Q):
    """Specify the data path"""
    path_factor = r"./database/factor"
    path_price  = r"./database/price"
    
    """Loading Data"""
    Factor = []
    Price = []
    factor_begin = 0
    factor_count = 0
    flag = 1
    for data_name in os.listdir(path_factor):
        if data_name == (start+".csv"):
            factor_begin = 1

        if factor_begin == 1:
            if flag % (F+1) == 1: # Load Factor
                x = pd.read_csv( path_factor + '/'+ data_name, header = None)
                Factor.append(x)
                factor_count += 1
            else:
                y = pd.read_csv( path_price + '/'+ data_name, header = None)
                Price.append(y)
                
            flag += 1
    
        if factor_count == sample_num+1:
            Factor = Factor[:-1]
            break
    
    
    """Data Regularization"""
    X_return = []
    Y_return = []
    for t in range( len(Factor) ):
        X = np.array( Factor[t], dtype=object )
        P = [ np.array( Price[i], dtype=object ) for i in range(F*t,F*t+F) ]
      
        # Step1: Find the stocks that appear in factor matrix and price vectors at the same time.
        n_x = set( X[:,0] )
        name_p = [ set(pp[:,0]) for pp in P]
        for n_p in name_p:
            n_x = n_p & n_x
        XX = []
        YY = []
        for row in X:
            if row[0] in n_x:
                XX.append( row[1:] )
                y = []
                for pp in P:
                    n_p = pp[:,0]
                    index = np.where( n_p == row[0] )[0][0]
                    y.append( pp[index][1] )
                y = np.array(y)
                YY.append( y )
            else:
                continue
        XX = np.array(XX, dtype = np.float)
        YY = np.array(YY, dtype = np.float)
        
        # Step2: Replace nan with column average
        XX = np.where(np.isnan(XX), np.ma.array(XX, mask=np.isnan(XX)).mean(axis=0), XX)
        
        # Step3: Factor Normalization
        for j in range( len( XX[0] ) ):
            max_j = max( XX[:,j] )
            min_j = min( XX[:,j] )
            for i in range( len( XX ) ):
                XX[i][j] = (XX[i][j]-min_j)/(max_j-min_j)
        
        # Step4: Standard Deviation; Return_Rate; Anomaly Filtering;
        Filter = [True for y in YY]
        return_std = []
        for i in range( len(YY) ):
            y = YY[i]
            mean = sum( y )/np.float( len(y) )
            std = sqrt( sum([ (price-mean)**2 for price in y])/np.float( len(y) ) )
            
            if std == 0:
                Filter[i] = False
                
            if y[0]!=0:
                return_rate = y[-1]/y[0]
            else:
                return_rate = np.nan
                Filter[i] = False
                
            return_std.append( ( return_rate, std) )
        
        XX = XX[ Filter ]
        return_std = np.array( return_std )
        return_std = return_std[ Filter ]
        YY = np.array( [ pair[0]/pair[1] for pair in return_std] )
        
        # Step5: Tail Set Construction
        XX = XX[ (-YY).argsort() ]
        for i in range( np.int( np.round( len(XX)*Q ) ) ):
            # Positive
            factor_positive = XX[i]
            X_return.append( factor_positive )
            Y_return.append( 1 )
            # Negative
            factor_negative = XX[-(i+1)]
            X_return.append( factor_negative )
            Y_return.append( 0 )
       
    # Shuffle
    X_return, Y_return = np.array( X_return ), np.array( Y_return )
    permutation = np.random.permutation(X_return.shape[0])
    X_return = X_return[permutation]
    Y_return = Y_return[permutation]

    return X_return, Y_return
