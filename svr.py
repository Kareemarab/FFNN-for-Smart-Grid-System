from io import StringIO
import requests
import json
import pandas as pd
import matplotlib.pyplot as pyplot

# @hidden_cell
# This function accesses a file in your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def get_object_storage_file_with_credentials_1eb260b6192e47c68ea8d791a45e2435(container, filename):
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
            'password': {'user': {'name': 'member_48c65902ff0758065a2ea68c9f74c92a5cedae94','domain': {'id': 'e2e8cfa27ebc46d589468fe63946471a'},
            'password': 'X1Qq{gTo&Z#5McSg'}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                        if(e2['interface']=='public'and e2['region']=='dallas'):
                            url2 = ''.join([e2['url'],'/', container, '/', filename])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO(resp2.text)

df = pd.read_csv(get_object_storage_file_with_credentials_1eb260b6192e47c68ea8d791a45e2435('DefaultProjectbluemixt6gmailcom', 'data.csv'))
df.head()

df['Time'] = pd.to_datetime(df['Time'])
df.index = df['Time']
del df['Time']

df.head()

df_Jan=df['2015-01':'2015-04']
df_Jan=df_Jan.resample('D', how='mean')

df_Jan['date'] = df_Jan.index
df_Jan['date'] = pd.to_datetime(df_Jan['date'])

import numpy as np
df_A=df_Jan.sort_values(by='temperature').values
X = df_A[:,0]
y = df_A[:,12]*1000
print(df_Jan.sort_values(by='temperature').values)

X=X.reshape(-1,1)
y=y.flatten()

#lastFew=-70
#X_train=X[:lastFew]
#X_test=X[lastFew:]
#y_train=y[:lastFew]
#y_test=y[lastFew:]

from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lm = LinearRegression()
svr_lin = SVR(kernel='linear', C=1)
svr_poly = SVR(kernel='poly', degree=2, coef0=5.0)
svr_rbf = SVR(kernel='rbf', gamma=0.1)
svr_rbf = SVR(kernel='rbf', gamma=0.1)

svr_rbf.fit(X_train, y_train)
svr_lin.fit(X_train, y_train)
svr_poly.fit(X_train, y_train)
lm.fit(X_train, y_train)

pred_SVRl = svr_lin.predict(X_test)
pred_SVRp = svr_poly.predict(X_test)
pred_SVRr = svr_rbf.predict(X_test)
pred_lm = lm.predict(X_test)

plt.figure(figsize=(20,5))
plt.plot(X,y,'m-', label='True data')
plt.plot(X_test,pred_SVRr, 'co',label='RBF model')
plt.plot(X_test,pred_SVRl,'g-',label='Linear model')
plt.plot(X_test,pred_SVRp,'bo', label='Polynomial model')
plt.plot(X_test, pred_lm, 'ro', label='Linear Reg')

plt.title('Support Vector Regression')
plt.legend(loc='upper right')

from sklearn.metrics import r2_score
print(r2_score(y_test, pred_SVRr))
print(r2_score(y_test, pred_SVRl))
print(r2_score(y_test, pred_SVRp))
print(r2_score(y_test, pred_lm))

from sklearn.metrics import explained_variance_score
print("Variance Score")
print(explained_variance_score(y_test, pred_SVRr))
print(explained_variance_score(y_test, pred_SVRl))
print(explained_variance_score(y_test, pred_SVRp))
print(explained_variance_score(y_test, pred_lm))

from sklearn.metrics import mean_squared_error
print("Mean Squared Error")
print(mean_squared_error(y_test, pred_SVRr))
print(mean_squared_error(y_test, pred_SVRl))
print(mean_squared_error(y_test, pred_SVRp))
print(mean_squared_error(y_test, pred_lm))

from sklearn.metrics import mean_squared_error
print("ACC")
print(1-((np.sum((np.absolute(y_test- pred_SVRr))/y_test))/len(y_test)))
print(1-((np.sum((np.absolute(y_test- pred_SVRl))/y_test))/len(y_test)))
print(1-((np.sum((np.absolute(y_test- pred_SVRp))/y_test))/len(y_test)))
print(1-((np.sum((np.absolute(y_test- pred_lm))/y_test))/len(y_test)))

errr  = np.absolute((y_test-pred_SVRr)/y_test) * 100
errr2 = np.sum(errr) / len(y_test)
errl  = np.absolute((y_test-pred_SVRl)/y_test) * 100
errl2 = np.sum(errl) / len(y_test)
errp  = np.absolute((y_test-pred_SVRp)/y_test) * 100
errp2 = np.sum(errp) / len(y_test)
errm  = np.absolute((y_test-pred_lm)/y_test) * 100
errm2 = np.sum(errm) / len(y_test)

acc1 = 100 - errr2
acc2 = 100 - errl2
acc3 = 100 - errp2
acc4 = 100 - errm2

print("Error for SVRr: ", "%.3f" % errr2, "%\n")
print("Accuracy for SVRr: ", "%.3f" % acc1, "%\n")
print("Error for SVRl: ", "%.3f" % errl2, "%\n")
print("Accuracy for SVRl: ", "%.3f" % acc2, "%\n")
print("Error for SVRp: ", "%.3f" % errp2, "%\n")
print("Accuracy for SVRp: ", "%.3f" % acc3, "%\n")
print("Error for SVRm: ", "%.3f" % errm2, "%\n")
print("Accuracy for SVRm: ", "%.3f" % acc4, "%\n")

plt.figure(figsize=(10,5))
plt.plot(y_test,pred_SVRr, 'co',label='RBF mode')
plt.plot(y_test,pred_SVRl,'go',label='Linear model')
plt.plot(y_test,pred_SVRp,'bo', label='Polynomial model')
plt.plot(y_test, pred_lm, 'ro', label='Linear Reg')
plt.title('Accuracy')
plt.legend(loc='upper right')
plt.ylim((1500,3500))
plt.xlim((1500,3500))

print(np.mean(np.abs((y_test - pred_lm) / y_test)) * 100)




