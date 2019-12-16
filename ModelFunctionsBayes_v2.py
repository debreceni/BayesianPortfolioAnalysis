# -*- coding: utf-8 -*-

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
import calendar
from datetime import date, timedelta
import scipy.optimize as sc
from sklearn.linear_model import LinearRegression as LR
from sklearn import preprocessing
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import pymc3 as pm
import theano
import logging
import random
logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

class getModelData():
    """
        Get the data needed for this model to run.  The data is sourced from Factset and stored in a CSV
        The Fama data is obtained from their website.
    """
    
    def getFamaData(self,startDt, endDt=None):

        #get the file
        factorurl = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
        resp = urlopen(factorurl)
        zipfile = ZipFile(BytesIO(resp.read()))
        #file name is F-F_Research_Data_Factors.CSV  Start at line 3
        with zipfile.open('F-F_Research_Data_Factors.CSV') as f:
            dfFama = pd.read_csv(f, header=2)

        momurl = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'
        resp = urlopen(momurl)
        zipfile = ZipFile(BytesIO(resp.read()))
        #file name is F-F_Research_Data_Factors.CSV  Start at line 3
        with zipfile.open('F-F_Momentum_Factor.CSV') as f:
            dfMomFama = pd.read_csv(f, header=11)

        dfFama.rename(columns={dfFama.columns[0]:"YearMonth"}, inplace=True)
        dfMomFama.rename(columns={dfMomFama.columns[0]:"YearMonth"}, inplace=True)

        dfFama.set_index('YearMonth', inplace=True)
        dfMomFama.set_index('YearMonth', inplace=True)

        dfFama = dfFama.loc[:' Annual Factors: January-December '][:-1]
        dfFama = dfFama.join(dfMomFama)
        dfFama.reset_index(inplace=True)
        dfFama['YearMonth'] = pd.to_datetime(dfFama['YearMonth'], format="%Y%m")
        dfFama = dfFama[dfFama['YearMonth']>=pd.to_datetime(startDt, format="%Y%m")]
        #convert to float
        cols = dfFama.columns.drop('YearMonth')
        dfFama[cols] = dfFama[cols].apply(pd.to_numeric,errors='coerce')
        #dfFama[cols] = dfFama[cols].apply(lambda x : (1+x/10000)**(12) -1)*100
        dfFama.columns = dfFama.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        dfFama.sort_values(by=['YearMonth'],inplace=True)
        return dfFama

    
    def getSecurityReturnData(self,filename):

        dfData = pd.read_csv(filename, header=0)
        dfData['Date'] = pd.to_datetime(dfData['Date'], format='%Y-%m-%d')

        lstcols = dfData.loc[:,dfData.isnull().values.all(axis=0)].columns.tolist()
        for col in lstcols:
            del dfData[col]
        cols = dfData.columns.drop('Date')
        #convert the data to being a true return value
        #dfData[cols] = dfData[cols].apply(lambda x : (1+x/10000)**(12) -1)*100
        dfData.sort_values(by=['Date'],inplace=True)
        return dfData, lstcols

    
    def getConstituentList(self, filename):

        dfData = pd.read_csv(filename, header=0)

        return dfData

    
    def getBenchmarkReturn(self,filename):
        
        #19991231
        dfData = pd.read_csv(filename, header=0)
        dfData['DATE'] = pd.to_datetime(dfData['DATE'], format='%Y%m%d')
        #add 1 month to get the begining of the next month
        dfData['MonthYear'] = [utilityFuncs.addsubMonth(pd.to_datetime(d.strftime('%Y%m'),format='%Y%m'),1).strftime('%Y%m') for d in dfData['DATE']]
        dfData['MonthYear'] = pd.to_datetime(dfData['MonthYear'], format="%Y%m")
        return dfData

    
    def getAllData(self, limit=None):
        
        #get downloaded security returns
        dfSecReturnData, removedcols = self.getSecurityReturnData(os.path.join(os.getcwd(), 'Data/securityReturns.csv'))
        dfSecReturnData.set_index('Date', inplace=True)
        #dfSecReturnData /= 100
        #Get the ticker list
        filename = os.path.join(os.getcwd(), 'Data/tickerEntity.csv')
        dfTickerList = self.getConstituentList(filename)
        #remove tickers that are no longer in use
        dfTickerList = dfTickerList[~dfTickerList['Entity'].isin(removedcols)]
       
        #limit the results
        if limit is not None:
            lstCols = random.sample(list(dfTickerList['Entity']),limit)
            dfTickerList = dfTickerList[dfTickerList['Entity'].isin(lstCols)]
            dfSecReturnData = dfSecReturnData.loc[:,lstCols]
            
        dfTickerList.set_index(['Entity','Ticker'], inplace=True)
        #remove date ranges not needed
        dfTickerList=dfTickerList.loc[:,dfSecReturnData.index.min().strftime('%Y-%m-%d'):dfSecReturnData.index.max().strftime('%Y-%m-%d')]
        
        #get the mininmum needed fama data
        dfFamaData  = self.getFamaData(dfSecReturnData.index.min())
        dfFamaData.set_index('YearMonth', inplace=True)
        #dfFamaData /= 100
        #need to check the fama data to see what the latest date available is
        maxDate = min([dfFamaData.index.max(),dfSecReturnData.index.max()])
        dfSecReturnData = dfSecReturnData.loc[:maxDate]
        
        dfBmk = self.getBenchmarkReturn(os.path.join(os.getcwd(), 'Data/R1000Rtn.csv'))
        dfBmk.set_index('MonthYear', inplace=True)
        dfBmk = dfBmk.loc[dfSecReturnData.index.min():dfSecReturnData.index.max(),:]
        #dfBmk['RETURN'] /= 100
        return dfSecReturnData, dfFamaData, dfTickerList, dfBmk

class modelFuncs():

    def portfolioMthlyPef(self,weights, m_returns, covMtx):

        rtn = np.dot(weights, m_returns)
        var = np.dot(np.dot(weights,covMtx),weights.T)
        return var, rtn


    def portVol(self,w,Q):

        return w.dot(Q).dot(w.T)

    def meanVar(self,w,mRtn, Q, lmb=0.01):

        #w.dot(mRtn) - lmb*(w.dot(Q).dot(w.T) + w.dot(D).dot(w.T))
        return 1/(w.dot(mRtn) - lmb*(w.dot(Q).dot(w.T)*2))
       # return (w.dot(mRtn)- rf )**2/(Qinv)

    def maxSharpeRatio(self,w, mRtn, rf, Q):

        var, rtn =self.portfolioMthlyPef(w,mRtn,Q)
        sr = (rtn - rf)/np.sqrt(var)
        return 1/sr

    def weightOptimizer(self, wFunc, num_assets,constraints, bound,  *args):

        const = constraints
        
        bounds = tuple(bound for asset in range(num_assets))

        optimized = sc.minimize(wFunc, num_assets*[1./num_assets,], args=args, method='SLSQP',bounds=bounds,constraints=const)
        #if not optimized.success: raise BaseException(optimized.message)
        return optimized
    
    def buildFactorCoef(self,dfSec,dfFact, rfRate, cols):
        def linRegessNext(X,Y):
            lr = LR()
            
            lr.fit(X[:-1,:],Y[1:,:])
            yHat = lr.predict(X[-1,:])
            err = (Y[-1,:]-yHat)**2
            return lr.coef_, lr.intercept_,err
        
        def linRegess(X,Y):
            lr = LR()
            lr.fit(X,Y)
            yHat = lr.predict(X)
            err = (Y-yHat).std()**2
            return lr.coef_, lr.intercept_,err        
        
        factBetas = np.zeros((len(cols), len(dfFact.columns)))
        factBetasNext = np.zeros((len(cols), len(dfFact.columns)))
        factIntercepts = np.zeros((len(cols)))
        errEst = np.zeros((len(cols)))
        #yHats = [] # np.zeros((len(cols),len(dfSec)))
        

        for idx, col in enumerate(cols):
            
            X = np.asmatrix(dfFact[dfFact.index.isin(dfSec[~dfSec[col].isnull()].index)].copy())
            curSec = np.asarray(dfSec[~dfSec[col].isnull()][col]).reshape(-1,1)
            curRF = np.asarray(rfRate[rfRate.index.isin(dfSec[~dfSec[col].isnull()].index)]).reshape(-1,1)
            
            Y = np.subtract(curSec,curRF)
            betasNext, interceptsNext, errNext = linRegessNext(X,Y)
            betas, intercepts, err = linRegess(X,Y)
            
            errEst[idx] = err

            factBetas[idx]=betas
            factBetasNext[idx]=betasNext
            factIntercepts[idx] = interceptsNext           
            

        return factBetas,factIntercepts, errEst, factBetasNext
    
    def buildBayesFactorCoef(self,dfSec,dfFact, rfRate, cols):
        
        def bayesRegressNext(X,Y, sampdraws):
            
            X_shared = theano.shared(X[:-1,:])
            Y_shared = theano.shared(Y[1:,:])
            
            with pm.Model() as normal_model:
                alpha = pm.Normal('alpha', mu=0, sd=1)
                beta = pm.Normal('beta', mu=0, sd=1, shape=(X.shape[1], 1))
                sigma = pm.HalfNormal('sigma', sd=10)

                # Expected value of outcome
                mu = alpha + pm.math.dot(X_shared, beta)
                # Define likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=Y_shared)
                # Inference!
                normal_trace = pm.sample(draws=sampdraws, chains = 2, tune = 500, progressbar = False, cores=20)
                
                
            with normal_model:
                #Set last value instead of using mean
                X_shared.set_value(X[-1:,:]) 
                post_pred = pm.sample_posterior_predictive(normal_trace, samples=500, progressbar = False)  
                
            return post_pred['likelihood'].mean(axis=0)
        
        def bayesRegress(X,Y, sampdraws):
            
            X_shared = theano.shared(X)
            Y_shared = theano.shared(Y)
            
            with pm.Model() as normal_model:
                alpha = pm.Normal('alpha', mu=0, sd=1)
                beta = pm.Normal('beta', mu=0, sd=1, shape=(X.shape[1], 1))
                sigma = pm.HalfNormal('sigma', sd=10)

                # Expected value of outcome
                mu = alpha + pm.math.dot(X_shared, beta)
                # Define likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=Y_shared)
                # Inference!
                normal_trace = pm.sample(draws=sampdraws, chains = 2, tune = 500, progressbar = False, cores=20)
                
                coef = normal_trace['beta'].mean(axis=0)
                err = normal_trace['sigma'].mean(axis=0)**2
                
                
            return coef, normal_trace['alpha'].mean(axis=0), err
            
        
        factBetas = np.zeros((len(cols), len(dfFact.columns)))
        factIntercepts = np.zeros((len(cols)))
        errEst = np.zeros((len(cols)))
        predRtns = np.zeros((len(cols)))
        
        for idx, col in enumerate(cols):
            
            X = np.asmatrix(dfFact[dfFact.index.isin(dfSec[~dfSec[col].isnull()].index)].copy())
            curSec = np.asarray(dfSec[~dfSec[col].isnull()][col])
            curRF = np.asarray(rfRate[rfRate.index.isin(dfSec[~dfSec[col].isnull()].index)])
            Y = np.subtract(curSec,curRF).reshape(-1,1)
            
            betas, intercept, err = bayesRegress(X,Y, 2000)
            predRtnNext = bayesRegressNext(X,Y, 2000)
            
            errEst[idx] = err

            factBetas[idx]=betas.reshape(4,)
            factIntercepts[idx] = intercept
            
            predRtns[idx] = predRtnNext.mean()           
            

        return factBetas,factIntercepts, errEst, predRtns



    def genIDXRtn(self,lstRtns):
        actRtns = [x[2] + 1 for x in lstRtns]
        estRtns = [x[3] + 1 for x in lstRtns]

        actualCumRtn = np.cumprod(actRtns) -1
        estCumRtn = np.cumprod(estRtns) - 1
        idxActual = (100 + actualCumRtn * 100).tolist()
        idxActual.insert(0,100)
        idxEst = (100 + estCumRtn * 100).tolist()
        idxEst.insert(0,100)

        return idxActual, idxEst
    
    def plotEffFrontier(Q, R):
                
        num_port = 25000
        results = np.zeros((3,num_port))
        
        for i in np.arange(num_port):
        
            weights = np.random.randint(100,size=len(R)) # np.random.rand(len(cols))
            weights = np.true_divide(weights, np.sum(weights))
            Prtn = np.dot(weights,R)
            Pstd = np.sqrt(np.dot(weights.T,np.dot(Q,weights)))
        
            results[0,i] = Prtn
            results[1,i] = Pstd
            results[2,i] = results[0,i]/results[1,i]
        
        results_frame = pd.DataFrame(results.T, columns=['ret','stdev','sharpe'])
        
        plt.scatter(results_frame.stdev,results_frame.ret,alpha=0.1)

class utilityFuncs():
    @staticmethod
    def addsubMonth( curDate, add_sub):

        for i in range(abs(add_sub)):
            daysinmonth = calendar.monthrange(curDate.year, curDate.month)[1]
            curDate = curDate + timedelta(days = daysinmonth)


        return curDate

    @staticmethod
    def MonthsBetweenDates( sDate, eDate):
        curDate = sDate
        mcnt = 0
        while (curDate < eDate):
            daysincurMonth = calendar.monthrange(curDate.year, curDate.month)[1]
            newDt = date(curDate.year, curDate.month,1)+ timedelta(days = daysincurMonth)
            daysinmonth = calendar.monthrange(newDt.year, newDt.month)[1]
            curDate = curDate + timedelta(days = daysinmonth)
            mcnt +=1

        return mcnt
