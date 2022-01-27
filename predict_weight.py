import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import NuSVR
import sklearn
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

def get_data():
    df_init = []

    with open("../BirthWeights/Nat2020_cut.txt", "r") as file:
        head = [next(file) for x in range(1000)]


    for line in head:

        new_row = {
                   'YEAR':      int(line[8:12]),
                   'M_Ht':      int(line[279:281]),  # if 99, no reporting
                   'M_Age':     int(line[74:76]),
                   'Lst_Prg':   int(line[280:281]),  # 88=FstPregnancy (cat)
                   'Wt_Gain':   int(line[305:306]),  # 9=unknown (cat)
                   'SEX':       line[474:475],
                   'GEST':      int(line[489:491]),
                   'BWt':       int(line[503:507]) # measured in grams
                   }

        if list(new_row.values())[1] != 99 and list(new_row.values())[6] != 99 and list(new_row.values())[7] != 9999 and list(new_row.values())[3] != ' ':
            df_init.append(new_row)

    return pd.DataFrame(df_init)

from sklearn.model_selection import train_test_split

def MSE(preds, true):
    mse = sum([(true[i] - preds[i])**2 for i in range(len(true))]) / len(true)
    return(mse)

df = get_data()

df['SEX'] = [0 if x =='F' else 1 for x in df['SEX']]
df['Wt_Gain'] = df['Wt_Gain'].astype("category")
df['Lst_Prg'] = df['Lst_Prg'].astype("category")

y = df['BWt']
X = df[['M_Ht', 'M_Age', 'Lst_Prg', 'Wt_Gain', 'SEX', 'GEST']]

dt = DecisionTreeRegressor(random_state=0, max_depth=5, min_samples_leaf=5, criterion='mse').fit(X, y)
lg = sm.OLS(y, sm.add_constant(X)).fit()
lda = LinearDiscriminantAnalysis().fit(X, y)
svm = make_pipeline(StandardScaler(), SVR(gamma='auto')).fit(X, y)
knn = KNeighborsRegressor(n_neighbors=5).fit(X, y)
nn = MLPRegressor(solver='sgd', max_iter=500, learning_rate='constant', activation='tanh').fit(X, y)
la = Lasso().fit(X, y)
ri = Ridge().fit(X, y)
gp = GaussianProcessRegressor(kernel=(DotProduct() + WhiteKernel())).fit(X, y)
nsvm = make_pipeline(StandardScaler(), NuSVR(C=.9, nu=.8)).fit(X, y)


dt_scores = cross_val_score(dt, X, y, cv = 5, scoring='neg_mean_squared_error')
print("DT Average Score = ", round(np.mean(dt_scores), 4))

lda_scores = cross_val_score(lda, X, y, cv = 5, scoring='neg_mean_squared_error')
print("LDA Average Score = ", round(np.mean(lda_scores), 4))

svm_scores = cross_val_score(svm, X, y, cv = 5, scoring='neg_mean_squared_error')
print("SVM Average Score = ", round(np.mean(svm_scores), 4))

knn_scores = cross_val_score(knn, X, y, cv = 5, scoring='neg_mean_squared_error')
print("KNN Average Score = ", round(np.mean(knn_scores), 4))

nn_scores = cross_val_score(nn, X, y, cv = 5, scoring='neg_mean_squared_error')
print("NN Average Score = ", round(np.mean(nn_scores), 4))

la_scores = cross_val_score(la, X, y, cv = 5, scoring='neg_mean_squared_error')
print("LA Average Score = ", round(np.mean(la_scores), 4))

ri_scores = cross_val_score(ri, X, y, cv = 5, scoring='neg_mean_squared_error')
print("RI Average Score = ", round(np.mean(ri_scores), 4))

gp_scores = cross_val_score(gp, X, y, cv = 5, scoring='neg_mean_squared_error')
print("GP Average Score = ", round(np.mean(gp_scores), 4))

nsvm_scores = cross_val_score(nsvm, X, y, cv = 5, scoring='neg_mean_squared_error')
print("NSVM Average Score = ", round(np.mean(nsvm_scores), 4))