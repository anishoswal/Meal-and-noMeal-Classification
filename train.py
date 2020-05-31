import pandas as pd
import pickle
from scipy import fftpack as fft
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    def movAvg(self):
        avg = pd.DataFrame(index=range(len(self.dataF)))
        avg = pd.concat([avg, self.dataF.mean(axis=1)], axis=1, ignore_index=True)
        return avg

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
        feature_matrix = pd.concat([self.FFT(), self.movAvg(), self.movStdError(), self.movKurtosis()], axis=1, ignore_index=True)
        return feature_matrix

    def features(file):
        df = DataMining.changeToDataFrame(file)
        patient = DataMining(df)
        patient.preprocess()
        mat = patient.matrix()
        return mat

    def principalComponentAnalysis(matrix):
        matrix = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=4)
        pca.fit(matrix)
        fname = open("PrincipalComponentAnalysis.pkl", 'wb')
        pickle.dump(pca, fname)
        fname.close()

    def Transformation(matrix, fname):
        file = open(fname, 'rb')
        pca = pickle.load(file)
        file.close()
        return pca.transform(matrix)

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

    def supportVectorMachine(X_train,X_test,y_train,y_test):
        print("----SUPPORT VECTOR MACHINE----")
        model=SVC(kernel='linear').fit(X_train,y_train)
        y_pred = model.predict(X_test)
        fname = open("SupportVectorMachine.pkl", 'wb')
        pickle.dump(model, fname)
        fname.close()
        print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
        print("Report : ",classification_report(y_test, y_pred))


mFeature1 = DataMining.features("data/mealData1.csv")
mFeature2 = DataMining.features("data/mealData2.csv")
mFeature3 = DataMining.features("data/mealData3.csv")
mFeature4 = DataMining.features("data/mealData4.csv")
mFeature5 = DataMining.features("data/mealData5.csv")
nmFeature1 = DataMining.features("data/Nomeal1.csv")
nmFeature2 = DataMining.features("data/Nomeal2.csv")
nmFeature3 = DataMining.features("data/Nomeal3.csv")
nmFeature4 = DataMining.features("data/Nomeal4.csv")
nmFeature5 = DataMining.features("data/Nomeal5.csv")

allFeatures = mFeature1.append([nmFeature1,mFeature2,nmFeature2,mFeature3,nmFeature3,mFeature4,nmFeature4,mFeature5,nmFeature5])
DataMining.principalComponentAnalysis(allFeatures)
mealD = pd.DataFrame(DataMining.Transformation(mFeature1.append([mFeature2,mFeature3,mFeature4,mFeature5]), "PrincipalComponentAnalysis.pkl"))
nomealD = pd.DataFrame(DataMining.Transformation(nmFeature1.append([nmFeature2,nmFeature3,nmFeature4,nmFeature5]), "PrincipalComponentAnalysis.pkl"))

mealD['class'] = 1
nomealD['class'] = 0
allFeatures = mealD.append([nomealD])

print("----K-FOLD CROSS VALIDATION----")
kfold = KFold(n_splits=6, random_state=7, shuffle=True)
for train_index, test_index in kfold.split(allFeatures):
    Training = allFeatures.iloc[train_index]
    Testing = allFeatures.iloc[test_index]
    X_train, y_train, X_test, y_test =  Training.loc[:,Training.columns!='class'],Training['class'],Testing.loc[:,Testing.columns!='class'],Testing['class']
    DataMining.supportVectorMachine(X_train, X_test, y_train, y_test)