import pandas as pd
import pickle
import sys
from scipy import fftpack as fft
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

class DataMining():
    def __init__(self, glucose):
        self.dataF = glucose

#preprocessing
    def preprocess(self):
        nan_row = self.dataF.isna().sum(axis=1)
        rowNaN_list = list()
        columns = len(self.dataF.iloc[0])
        for i in range(len(nan_row)):
            if nan_row.iloc[i] > 0.4 * columns:
                rowNaN_list.append(i)
        self.dataF.drop(rowNaN_list, inplace=True)
        self.dataF.reset_index(inplace=True, drop=True)
        self.dataF.interpolate(method='quadratic', order=2, inplace=True)
        self.dataF.bfill(inplace=True)
        self.dataF.ffill(inplace=True)

#fast fourier transform
    def FFT(self):
        fftG = fft.rfft(self.dataF, n=5, axis=1)
        fftGr = pd.DataFrame(data=fftG)
        return fftGr


#moving average
    def cgmVelocity(self):
        cgmV = pd.DataFrame(index=range(len(self.dataF)))
        cgmV = pd.concat([cgmV, self.dataF.mean(axis=1)], axis=1, ignore_index=True)
        return cgmV

#moving standard error of mean
    def movStdError(self):
        count = 0
        stdErr = pd.DataFrame(index=range(len(self.dataF)))
        while count < len(self.dataF.loc[0]) // 5 - 1:
            stdErr = pd.concat([stdErr, self.dataF.iloc[:, 0 + (4 * count):10 + (4 * count)].sem(axis=1)], axis=1, ignore_index=True)
            count += 1
        return stdErr

#moving kurtosis
    def movKurtosis(self):
        count = 0
        kurto = pd.DataFrame(index=range(len(self.dataF)))
        while count < len(self.dataF.loc[0]) // 5 - 1:
            kurto = pd.concat([kurto, self.dataF.iloc[:, 0 + (4 * count):10 + (4 * count)].kurtosis(axis=1)], axis=1, ignore_index=True)
            count += 1
        return kurto

    def matrix(self):
        feature_matrix = pd.concat([self.FFT(), self.cgmVelocity(), self.movStdError(), self.movKurtosis()], axis=1, ignore_index=True)
        return feature_matrix

    def changeToDataFrame(file):
        dFrame = pd.DataFrame()
        for row in open(file):
            row_values = row.strip().split(',')
            length = len(row_values)
            for i in range(length):
                row_values[i] = float(row_values[i])
            npArray = np.array(row_values)
            leng = len(npArray)
            if leng <= 30:
                for i in range(leng,30):
                    npArray = np.append(npArray,np.nan)
            else:
                npArray = npArray[:][:30]
            dFrame = dFrame.append(pd.DataFrame(npArray.reshape((1,30))), ignore_index=True)
        return dFrame

    def Transformation(matrix, fname):
        file = open(fname, 'rb')
        pca = pickle.load(file)
        file.close()
        return pca.transform(matrix)

def arguments(argv):
    p = ArgumentParser(description="Meal or No-meal Classification")
    p.add_argument("--file","-f",type=str, required= True,help='Test file name/address')
    return p.parse_args(argv)

argz = arguments(sys.argv[1:])
test = argz.file
testDf = DataMining.changeToDataFrame(test)
t = DataMining(testDf)
t.preprocess()
features = t.matrix()
features = DataMining.Transformation(features, "PrincipalComponentAnalysis.pkl")
r,_= testDf.shape
y_pred = [1 for i in range(r)]

model = pickle.load(open("SupportVectorMachine.pkl", 'rb'))
y_test = model.predict(features)
acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred) * 100
rec = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
print("-------------------------------PREDICTED OUTPUT-------------------------------\n%s"%(y_test))
print("\n-------------------------------------------------------------------------------------------")
print("|| DISCLAIMER: Accuracy is shown assuming that all of the data belongs to the MEAL Class ||")
print("\n-------------------------------------------------------------------------------------------------------------------")
print("|||| Accuracy: {0} || Precision: {1} || Recall:{2} || F1-Score:{3} ||||".format(acc, prec, rec, f1))
print("-------------------------------------------------------------------------------------------------------------------\n")