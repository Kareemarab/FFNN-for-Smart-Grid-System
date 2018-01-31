import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import sys

from datetime import datetime
from scipy import optimize
from sklearn.metrics import r2_score, classification_report, mean_squared_error

### Methods ###

### properties and constants ###
np.set_printoptions(suppress=True)

### Data extraction from csv file ###
times = pd.read_csv("full_time_only.csv")
data = pd.read_csv("data.csv")
### Data extraction from csv file ###
data.columns = ["Time", "temperature", "icon", "humidity", "visibility", "summary", "apparentTemperature", "pressure", "windSpeed", "epochTime", "windBearing", "precipIntensity", "dewPoint", "precipProbability", "Apartment1",  "Apartment2",  "Apartment3",  "Apartment4",  "Apartment5"]
time = list(data.Time)
temp = list(data.temperature)
humi = list(data.humidity)
visi = list(data.visibility)
atmp = list(data.apparentTemperature)
pres = list(data.pressure)
wspd = list(data.windSpeed)
etim = list(data.epochTime)
wbrg = list(data.windBearing)
prei = list(data.precipIntensity)
dpnt = list(data.dewPoint)
prep = list(data.precipProbability)
apt1 = list(data.Apartment1)
apt2 = list(data.Apartment2)
apt3 = list(data.Apartment3)
apt4 = list(data.Apartment4)
apt5 = list(data.Apartment5)

print(time)
'''
### Training data ###
# Inputs
time_tr = times[0:6128]
temp_tr = temp[0:6128]
humi_tr = humi[0:6128]
visi_tr = visi[0:7291]
atmp_tr = atmp[0:7291]
pres_tr = pres[0:7291]
wspd_tr = wspd[0:7291]
etim_tr = etim[0:7291]
wbrg_tr = wbrg[0:7291]
prei_tr = prei[0:7291]
dpnt_tr = dpnt[0:7291]
prep_tr = prep[0:7291]
# Outputs
apt1_tr = apt1[0:6128]
apt2_tr = apt2[0:2155]
apt3_tr = apt3[0:2155]
apt4_tr = apt4[0:2155]
apt5_tr = apt5[0:2155]

### Testing data ###
# Inputs
time_te = times[6129:8755]
temp_te = temp[6129:8755]
humi_te = humi[6129:8755]
visi_te = visi[7292:7459]
atmp_te = atmp[7292:7459]
pres_te = pres[7292:7459]
wspd_te = wspd[7292:7459]
etim_te = etim[7292:7459]
wbrg_te = wbrg[7292:7459]
prei_te = prei[7292:7459]
dpnt_te = dpnt[7292:7459]
prep_te = prep[7292:7459]
# Outputs
apt1_te = apt1[6129:8755]
apt2_te = apt2[2156:2323]
apt3_te = apt3[2156:2323]
apt4_te = apt4[2156:2323]
apt5_te = apt5[2156:2323]

### Time data evaluations ###

### Data normalization ###
time_tr = np.divide(time_tr, 100)
temp_tr = np.divide(temp_tr, 100)
humi_tr = np.divide(humi_tr, 100)
visi_tr = np.divide(visi_tr, 100)
atmp_tr = np.divide(atmp_tr, 100)
pres_tr = np.divide(pres_tr, 187)
wspd_tr = np.divide(wspd_tr, 100)
etim_tr = np.divide(etim_tr, 100)
wbrg_tr = np.divide(wbrg_tr, 187)
prei_tr = np.divide(prei_tr, 0.1)
dpnt_tr = np.divide(dpnt_tr, 100)
prep_tr = np.divide(prep_tr, 100)
apt1_tr = np.divide(apt1_tr, 100)
apt2_tr = np.divide(apt2_tr, 100)
apt3_tr = np.divide(apt3_tr, 100)
apt4_tr = np.divide(apt4_tr, 100)
apt5_tr = np.divide(apt5_tr, 100)

time_te = np.divide(time_te, 100)
temp_te = np.divide(temp_te, 100)
humi_te = np.divide(humi_te, 100)
visi_te = np.divide(visi_te, 100)
atmp_te = np.divide(atmp_te, 100)
pres_te = np.divide(pres_te, 187)
wspd_te = np.divide(wspd_te, 100)
etim_tr = np.divide(etim_tr, 100)
wbrg_tr = np.divide(wbrg_tr, 187)
prei_tr = np.divide(prei_tr, 0.1)
dpnt_tr = np.divide(dpnt_tr, 100)
prep_tr = np.divide(prep_tr, 100)
apt1_te = np.divide(apt1_te, 100)
apt2_te = np.divide(apt2_te, 100)
apt3_te = np.divide(apt3_te, 100)
apt4_te = np.divide(apt4_te, 100)
apt5_te = np.divide(apt5_te, 100)

### Dimensionality expansion ###
# Inputs
temp_tr_exp = np.expand_dims(temp_tr, axis=1)
humi_tr_exp = np.expand_dims(humi_tr, axis=1)
visi_tr_exp = np.expand_dims(visi_tr, axis=1)
atmp_tr_exp = np.expand_dims(atmp_tr, axis=1)
pres_tr_exp = np.expand_dims(pres_tr, axis=1)
wspd_tr_exp = np.expand_dims(wspd_tr, axis=1)
etim_tr_exp = np.expand_dims(etim_tr, axis=1)
wbrg_tr_exp = np.expand_dims(wbrg_tr, axis=1)
prei_tr_exp = np.expand_dims(prei_tr, axis=1)
dpnt_tr_exp = np.expand_dims(dpnt_tr, axis=1)
prep_tr_exp = np.expand_dims(prep_tr, axis=1)

temp_te_exp = np.expand_dims(temp_te, axis=1)
humi_te_exp = np.expand_dims(humi_te, axis=1)
visi_te_exp = np.expand_dims(visi_te, axis=1)
atmp_te_exp = np.expand_dims(atmp_te, axis=1)
pres_te_exp = np.expand_dims(pres_te, axis=1)
wspd_te_exp = np.expand_dims(wspd_te, axis=1)
etim_te_exp = np.expand_dims(etim_te, axis=1)
wbrg_te_exp = np.expand_dims(wbrg_te, axis=1)
prei_te_exp = np.expand_dims(prei_te, axis=1)
dpnt_te_exp = np.expand_dims(dpnt_te, axis=1)
prep_te_exp = np.expand_dims(prep_te, axis=1)

# Outputs
apt1_tr_exp = np.expand_dims(apt1_tr, axis=1)
apt2_tr_exp = np.expand_dims(apt2_tr, axis=1)
apt3_tr_exp = np.expand_dims(apt3_tr, axis=1)
apt4_tr_exp = np.expand_dims(apt4_tr, axis=1)
apt5_tr_exp = np.expand_dims(apt5_tr, axis=1)
apt1_te_exp = np.expand_dims(apt1_te, axis=1)
apt2_te_exp = np.expand_dims(apt2_te, axis=1)
apt3_te_exp = np.expand_dims(apt3_te, axis=1)
apt4_te_exp = np.expand_dims(apt4_te, axis=1)
apt5_te_exp = np.expand_dims(apt5_te, axis=1)

### Cleaning up ###
X  = np.column_stack((time_tr,temp_tr_exp,humi_tr_exp))
Xt = np.column_stack((time_te,temp_te_exp,humi_te_exp))
Xt2 = np.column_stack((time_te,temp_te_exp,humi_te_exp))
Xt3 = np.column_stack((time_te,temp_te_exp,humi_te_exp))
Xt4 = np.column_stack((time_te,temp_te_exp,humi_te_exp))
Xt5 = np.column_stack((time_te,temp_te_exp,humi_te_exp))
y  = apt1_tr_exp
yt = apt1_te_exp

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 50
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2  
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad   
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


NN = Neural_Network()
tr = trainer(NN)
print('Training neural network...')
tr.train(X,y)
ynn = NN.forward(X)
yn = NN.forward(Xt)

np.savetxt('pred.txt', yn, delimiter=", ",fmt='%.6f')
np.savetxt('real.txt', y, delimiter=", ",fmt='%.6f')
t = np.linspace(0,109.5,len(yn[::30]))

#plt.subplot(211)

plt.figure(figsize=(17.37,7.25))
plt.rcParams.update({'font.size': 30})
plt.plot(t, yn[::30]*100000, '^-', label="FFNN (prediction)", ms=13)
plt.plot(t, apt1_te[::30]*100000, 'm-', linewidth=3, label="True data")
plt.xlabel('Days', fontsize=30)
plt.ylabel('Electricity (Watt-hour)', fontsize=30)
plt.legend(loc='upper left', fontsize=27)
plt.axis([0,109.5,-500,5000])
#plt.savefig('6c.pdf', bbox_inches='tight')
#plt.subplot(212)
#plt.plot(yn, color='blue', label="Pred")
#plt.plot(apt1_te, color='red', label="Actual")
#plt.legend(loc='upper right')
#plt.title('Actual vs. Predicted (TESTING)')

hr = np.sum(np.absolute(yn-yt))
hy = np.sum(np.absolute(yt))

hi = hr/hy

e  = np.absolute((yn-yt)/yt) / len(time_te)
e*100
err  = np.sum(e)
acc = 100 - err

r2 = r2_score(yt,yn)
summ  = np.sum(pow((yn-yt),2))
rmsd  = np.sqrt(summ/len(time_te))
nrmsd = (rmsd/(np.amax(yt)-np.amin(yt))) * 100
f = 100 - nrmsd
plt.savefig('test22.pdf', bbox_inches='tight')
print("Feed-Forward Neural Network: ")
print("\nr2_Score:", "%.3f" % acc)
print("RMSE:", "%.3f" % f, "%\n")
print(hi)

plt.show()
'''