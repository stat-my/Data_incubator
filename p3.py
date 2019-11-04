#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:07:36 2019

@author: lmy
"""


#Helper function
import csv
import os
os.chdir('../Data')

class Diagnosis:
    def __init__(self, TYPE):	
        self.patientDiagnosis = {}
        #self.diagArr = ['272.2', '401.1']
        with open(TYPE+'Set/'+TYPE+'_SyncDiagnosis.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) 
            for DiagnosisGuid,	PatientGuid,	ICD9Code	, DiagnosisDescription, 	StartYear, 	StopYear	, Acute, 	UserGuid in reader:
                if not PatientGuid in self.patientDiagnosis:
                    self.patientDiagnosis[PatientGuid]=[0,0]
                if ICD9Code=='272.2':
                    self.patientDiagnosis[PatientGuid][0]=1
                elif ICD9Code=='401.1':
                    self.patientDiagnosis[PatientGuid][1]=1
                else:
                    pass
    def getDiagnosis(self, patientGUID):
        if not self.patientDiagnosis.get(patientGUID): 
            return [0,0]
        else:    
            return self.patientDiagnosis.get(patientGUID)
        
class Medicine:
    def __init__(self, TYPE):	
        self.patientMedicine = {}
        #self.diagArr = ['272.2', '401.1']
        with open(TYPE+'Set/'+TYPE+'_SyncMedication.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) 
            for l in reader:
                PatientGuid = l[1]
                if not PatientGuid in self.patientMedicine:
                    self.patientMedicine[PatientGuid]=[0,0,0,0,0]
                if l[3]=='Lisinopril oral tablet':
                    self.patientMedicine[PatientGuid][0]=1
                elif l[3]=='Lipitor (atorvastatin) oral tablet':
                    self.patientMedicine[PatientGuid][1]=1
                elif l[3]=='Simvastatin oral tablet':
                    self.patientMedicine[PatientGuid][2]=1
                elif l[3]=='Zithromax Z-Pak (azithromycin) oral tablet':
                    self.patientMedicine[PatientGuid][3]=1
                elif l[3]=='Xanax (ALPRAZolam) oral tablet':
                    self.patientMedicine[PatientGuid][4]=1
                else:
                    pass
    def getMedicine(self, patientGUID):
        if not self.patientMedicine.get(patientGUID): 
            return [0,0,0,0,0]
        else:    
            return self.patientMedicine.get(patientGUID)

import pandas as pd
obs = pd.read_csv('trainingSet/training_SyncLabObservation.csv')
panel = pd.read_csv('trainingSet/training_SyncLabPanel.csv')
res = pd.read_csv('trainingSet/training_SyncLabResult.csv')

obs_panel = obs.merge(panel, on='LabPanelGuid')
obs_panel_res = obs_panel.merge(res, on='LabResultGuid')
obs_panel_res.to_csv('trainingSet/training_SyncLabObservationFULL.csv')


obs = pd.read_csv('testSet/test_SyncLabObservation.csv')
panel = pd.read_csv('testSet/test_SyncLabPanel.csv')
res = pd.read_csv('testSet/test_SyncLabResult.csv')

obs_panel = obs.merge(panel, on='LabPanelGuid')
obs_panel_res = obs_panel.merge(res, on='LabResultGuid')
obs_panel_res.to_csv('testSet/test_SyncLabObservationFULL.csv')

       
class LabObservation:
    def __init__(self, TYPE):	
        self.patientLabObservation = {}
        #self.diagArr = ['272.2', '401.1']
        with open(TYPE+'Set/'+TYPE+'_SyncLabObservationFULL.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) 
            
            for l in reader:
                PatientGuid = l[18]
                if not PatientGuid in self.patientLabObservation:
                    self.patientLabObservation[PatientGuid]=[0,0,0]
                if l[1]=='Bilirubin':
                    self.patientLabObservation[PatientGuid][0]=1
                elif l[1]=='Protein':
                    self.patientLabObservation[PatientGuid][1]=1
                elif l[1]=='Hemoglobin':
                    self.patientLabObservation[PatientGuid][2]=1
                else:
                    pass
    def getLabObservation(self, patientGUID):
        if not self.patientLabObservation.get(patientGUID): 
            return [0,0,0]
        else:    
            return self.patientLabObservation.get(patientGUID)

  
    
class Patient:
	def __init__(self, csvfileReaderRow):	
		row = csvfileReaderRow
		self.PatientGuid = row[0]
		self.DoctorGuid = row[5]
		
		self.featureVector = []
		
		#Do they have diabetes
		self.hasDiabetes = (int(row[1]))
		
		#Gender Test, 1 for Male, 0 for female
		if row[2] == "M": self.featureVector.append(1)
		else: self.featureVector.append(0)
		
		#Age test, see AgeGroup for details
		self.featureVector.append(AgeGroup(int(row[3])))
		
		
		#Region test, see US_Region for details
		self.featureVector+=US_Region(row[4])
	
	def addFeatures(self, features):
		self.featureVector += features
	
#Gets the age group by decade a patient was born in
def AgeGroup(year_of_birth):
	dec = year_of_birth - 1900
	if dec < 10: return 0
	if dec < 20: return 1
	if dec < 30: return 2
	if dec < 40: return 3
	if dec < 50: return 4
	if dec < 60: return 5
	if dec < 70: return 6
	if dec < 80: return 7
	if dec < 90: return 8
	if dec < 100: return 9

def US_Region(state):
	WEST = ['WA', 'OR', 'WY', 'MT', 'ID', 'CO', 'UT', 'NV', 'AZ', 'CA','NM', 'AK', 'HI']
	MIDWEST = ['ND', 'SD', 'MN', 'WI', 'MI', 'NE', 'IA', 'IL', 'IN', 'OH', 'KS', 'MO']
	SOUTH = ['TX', 'OK', 'AR', 'LA', 'WV', 'MD', 'DE', 'DC', 'KY', 'VA', 'TN', 'NC', 'MS', 'AL', 'GA', 'SC', 'FL', 'PR']
	NORTHEAST = ['ME', 'NH', 'VT', 'NY', 'MA', 'RI', 'CT', 'NJ', 'PA']

	if state in WEST: return [1,0,0,0]
	if state in MIDWEST: return [0,1,0,0]
	if state in SOUTH: return [0,0,1,0]
	if state in NORTHEAST: return [0,0,0,1]
	raise Exception(state + " not found in any region")



def getPatients(TYPE):
	with open(TYPE+'Set/'+TYPE+'_SyncPatient.csv', 'r+') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)         
		
		#Create the patients 
		patients = []
		for row in reader:
			patients.append(Patient(row))
		results = addMoreData(patients, TYPE)

	return results

def addMoreData(patients, TYPE):
    results = []
    med = Medicine(TYPE)
    diag = Diagnosis(TYPE)
    obs = LabObservation(TYPE)
    for p in patients:
        m = med.getMedicine(p.PatientGuid)
        d = diag.getDiagnosis(p.PatientGuid)
        o = obs.getLabObservation(p.PatientGuid)
        p.addFeatures(m)
        p.addFeatures(d)
        p.addFeatures(o)
        results.append(p)
    return results

#train/test data preparation

import numpy as np
data = getPatients('training')
data = np.array([[data[i].featureVector+[data[i].hasDiabetes] for i in range(len(data))]]).reshape(-1, 17)

from sklearn.model_selection import train_test_split


logi_error=[]
rf_error = []
gbr_error = []
for t in range(20):
    print(t)


    train, test = train_test_split(data , test_size=0.5)
    traindf = pd.DataFrame(train)
    testdf = pd.DataFrame(test)
    from sklearn.metrics import mean_squared_error
    ytrue = pd.np.array(testdf.iloc[:,-1],dtype=float)
    
    
    
    
    #logistic regression training and testing
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(traindf.iloc[:,0:(-1)], traindf.iloc[:,-1])
    yhat = clf.predict(testdf.iloc[:,0:(-1)])
    logi_error.append(mean_squared_error(ytrue, pd.np.array(yhat,dtype=float)))
    #0.19798994974874373
    
    
    #GBR training and testing
    from sklearn.ensemble import GradientBoostingRegressor
    params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
    clf = GradientBoostingRegressor(**params)
    clf.fit(traindf.iloc[:,0:(-1)], traindf.iloc[:,-1])
    yhat = clf.predict(testdf.iloc[:,0:(-1)])
    gbr_error.append(mean_squared_error(ytrue, pd.np.array(yhat,dtype=float)))
    #0.1597967334288979
    
    
    #random forest training and testing
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(traindf.iloc[:,0:(-1)], traindf.iloc[:,-1])
    yhat = clf.predict(testdf.iloc[:,0:(-1)])
    rf_error.append(mean_squared_error(ytrue, pd.np.array(yhat,dtype=float)))
    #0.14691153729211223





import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt 


## combine these different collections into a list    
data_to_plot = [logi_error,gbr_error,rf_error ]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)
ax.set_xticklabels(['Logistic Regression', 'Gradient Boost', 'Random Forest'])
ax.set_ylabel('Mean Squared Error')
fig.suptitle('Comparision of Baseline Models for Predicting Diabetes', fontsize=14, fontweight='bold')




from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100)
clf.fit(traindf.iloc[:,0:(-1)], traindf.iloc[:,-1])
yhat = clf.predict(testdf.iloc[:,0:(-1)])
gbr_error.append(mean_squared_error(ytrue, pd.np.array(yhat,dtype=float)))


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)





