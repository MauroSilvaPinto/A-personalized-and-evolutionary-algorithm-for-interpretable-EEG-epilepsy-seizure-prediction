"""
Database class

This class should be changed, as it is not totally intuitive.
In other words, it combines functions which should be in different classes.
Nevertheless, it is fully functional as all the code.

Its main functions:
    - handle patients to select/remove them, but this should be removed in
      the future as this algorithm turned out to be patient-specific
    - surrogate analysis (statistical validation)
    - fitness function (with past (using lag features) and without past (without
      using lag features))
    - calculate predictive power of each hyper-feature
    
These functions (predictive power, surrogate analysis, and fitness functions)
contain a lot of redundancy inside, as these were changed several times.
I'm sorry about having so much code redundance in this script

"""


import os
from Patient import Patient
import numpy as np
from Classifier import Classifier
from Filter import Filter
import math
import scipy
import random
#from statsmodels.tsa.api import VAR, DynamicVAR

class Database:

    # initializes the database with the provided directory
    # loads the patient id list
    # loads the seizure data from all patients respectively
    def __init__(self, path):
        self.directory=path
        
        self.patient_list=[]
        self.loadPatientList()
        
        self.patients=[]
        self.loadAllPatients()
        
    
    # loads the patient list by iterating all files in the directory
    def loadPatientList(self):
        filenames=self.getFilenames()
        
        for i in range(len(filenames)):
            if  Database.isFilenameSeizureData(filenames[i]):
                patient_number=Database.getPatientNumberFromFilename(filenames[i])    
                if (patient_number not in self.patient_list and 
                    not "_pre" in patient_number and
                    not patient_number == "pat"):
                    self.addPatientToList(patient_number)  
     
              
    
    # iterates the function loadPatient for the whole patients list         
    def loadAllPatients(self):
        for i in range(len(self.patient_list)):
            self.loadPatient(self.patient_list[i])
            
    
    # loads the patient data of the provided number into the database
    def loadPatient(self,number):
        new_patient=Patient(number, self.directory)
        self.patients.append(new_patient)
    
    # eliminates all patients from the database and the only one provided by 
    # the patient number    
    def eliminateAllPatientsExcept(self, patient_number):
        self.patient_list=patient_number
        self.patients=[]
        self.loadPatient(patient_number)
        
    def eliminatePatientsWithLowSeizureNumber(self,minimum_seizures):
        for patient in reversed(self.patients):
            if len(patient.seizures_data)<minimum_seizures:
                self.patients.remove(patient)
                self.patient_list.remove(str(patient.patient_number))
            
        
    # confirms if a name file contains seizure data, that is, if it contains
    # the preictal expression in its name
    def isFilenameSeizureData(name):
        return ("preictal" in name)
    
    # from the filename, retrieves the number of the patient
    def getPatientNumberFromFilename(name):
        return name.split('_')[2]
    
    # adds a patient number to the list of patients
    def addPatientToList(self,patient_number):
        self.patient_list.append(patient_number)
       
    # retrieves the filenames of the directory provided to build the
    # database    
    def getFilenames(self):
        os.chdir(self.directory)
        filenames=os.listdir()    
        return filenames

    # prints the patients list, that is, their id numbers
    def printPatientList(self):
        print(self.patient_list)
        
    # retrieves a patient by providing its id number    
    def getPatientFromNumber(self, number):
        for i in range(len(self.patients)):
            if int(self.patients[i].getNumber())==int(number):
                return self.patients[i]
    
    # prints the list of seizures of a certain patient, by providing its
    # id number    
    def printSeizuresFromPatient(self, number):
        patient=self.getPatientFromNumber(number)
        patient.printSeizures()
        
    # prints the id and respective seizures of all patients present
    # in the database
    def printSeizuresFromAllPatients(self):
        for patient in self.patients:
            print("Patient: " + patient.getNumber())
            self.printSeizuresFromPatient(patient.getNumber())
        
    
    # returns the duration in samples of all seizures of all patients as an
    # numpy array
    def getSeizuresDurationInSamples(self):
        seizures_duration=[]
        for patient in self.patients:
           seizures_duration=seizures_duration+patient.getSeizuresDurationInSamples()
        return np.array(seizures_duration)
            
    # returns the duration in minutes of all seizures of all patients as an
    # numpy array
    def getSeizuresDurationInMinutes(self):
        return self.getSeizuresDurationInSamples()*5/60
            
        
    # eliminates the seizures from the database with less than the provided minutes   
    def eliminateSeizuresWithLessThan(self,minutes):
        for patient in reversed(self.patients):
            patient.eliminateSeizuresWithLessThan(minutes)
            if not Database.hasPatientEnoughSeizures(len(patient.seizures_data)):
                self.patients.remove(patient)
                self.patient_list.remove(str(patient.patient_number))
    
    # confirm if a given patient has enough patients to be evaluated, that is,
    # it has at least 3, one for train, one for validate and another for testing
    def hasPatientEnoughSeizures(number_seizures):
        return number_seizures>=3

    

    
        #apply the filter to the database to obtain the fitness function, that is:
    #
    #   for each patient:
    #       selects the training data
    #       pre-processes the data (remove NaN, Inf, etc, label balancement, zScoring)
    #       trains a classifier
    #       trains a post-processing threshold with a regression classifier
    #
    #       selects the validating data
    #       applies the same preprocessing (remove Nan, Inf, etc, applies the
    #                                       zScore of the training with training
    #                                       mean and training std)
    #       applies the classifier trained
    #       applies the post-processing regressiont trained
    #       
    #       with the obtained labels, calculate the AUC of the ROC curve for each patient
    #
    # returns the mean roc curve       
    def applyFilterToDatabase(self,filter,objective,metrics,method_metrics,sop_period):   
        preictal_time, filter_size, filter_components=Filter.decomposeFilter(filter)
        if sop_period==0:
            sop_period=preictal_time
        else:
            sop_period=sop_period
            preictal_time=sop_period
           
        for patient in self.patients:            
            
            patient_number=patient.patient_number
            roc_curves=[]
            sensibility=[]
            false_positive_rates=[]
            false_positive_hour_rates=[]
            fitness=[]
            
            number_of_seizures=[]
            number_of_triggered_seizures=[]
            number_of_false_alarms=[]
            ratio_penalties=[]
            
            
            # getting seizures
            if objective == "validation_data":
                validation_stage="train"
            elif objective == "test_data":
                validation_stage="test"
                
            # get seizure indexes (for training or for testing)
            seizures_indexes=patient.getSeizureIndexesFor(validation_stage)
            
            # iterating the selected seizures
            for index in range(0,len(seizures_indexes)-1):
                                
                #################### training procedure ####################
                
                # training seizure
                training_data=patient.getSeizureIndexData(seizures_indexes[index],
                                                          filter_size, 
                                                          preictal_time, 
                                                          filter_components,
                                                          validation_stage)
                                 
                ## Splitting into labels and features
                training_labels=training_data[:,-1]
                training_data=training_data[:,:-1]
                
                
                ########################### Lagging ###########################

                # constructing the feature delays
                delay_units=3
                # features with lags
                recurrent_training_data=np.zeros((len(training_labels)-delay_units,
                                                    np.shape(training_data)[1]*delay_units))              
                # labels with lags
                recurrent_training_labels=np.zeros(len(training_labels)-delay_units)
                
            
                # retrieving the lags
                for iii in range(delay_units-1,len(training_labels)-1):
                    for delay in range(0,delay_units):
                        recurrent_training_data[iii-delay_units+1,np.arange(0,np.shape(training_data)[1])+np.shape(training_data)[1]*delay]=training_data[iii-delay,:]
                    recurrent_training_labels[iii-delay_units+1]=training_labels[iii]
                    
                
                ## renaming the training_data and training_labels for the lag ones
                training_labels=recurrent_training_labels
                training_data=recurrent_training_data
                               
                ########################### Lagging ###########################
                
                
                #Preprocessing the data            
                training_data,mean,std=Classifier.preProcessTrainingData(training_data)
                
                redundant_features=[]
                #get redundant feature with a threshold higher than 0.95
                for feature_index in range(np.shape(training_data)[1]-1,-1,-1):
                    for feature_index_j in range(np.shape(training_data)[1]-2,0,-1):
                        if feature_index==feature_index_j:
                            continue
                        elif ((feature_index in redundant_features) or 
                              (feature_index_j in redundant_features)):
                            continue
                        else:
                            feature_a=training_data[:,feature_index]
                            feature_b=training_data[:,feature_index_j]
                            if abs(np.corrcoef(feature_a,feature_b)[0][1])>0.95:
                                redundant_features.append(feature_index)
                
                # create a vector with non-redundant features
                good_features_indexes=np.delete(np.arange(0,np.shape(training_data)[1]),
                                                redundant_features)
                # selecting the non-redundant features only
                training_data=training_data[:,good_features_indexes]                        
                
                                
                # First Phase training
                classifier=Classifier.trainClassifier(training_data,training_labels,"over")
                                
                
                #################### validation procedure ####################
                
                # testing seizure
                validation_data=patient.getSeizureIndexData(seizures_indexes[index+1],
                                                          filter_size, 
                                                          sop_period, 
                                                          filter_components,
                                                          validation_stage)
                 
                
                ## Splitting into labels and features
                validation_labels=validation_data[:,-1]
                validation_data=validation_data[:,:-1]
                
                 ########################### Lagging ###########################

                # features with lags
                recurrent_validation_data=np.zeros((len(validation_labels)-delay_units,
                                                    np.shape(validation_data)[1]*delay_units))              
                # labels with lags
                recurrent_validating_labels=np.zeros(len(validation_labels)-delay_units)
                
                # retrieving the lags
                for iii in range(delay_units-1,len(validation_labels)-1):
                    for delay in range(0,delay_units):
                        recurrent_validation_data[iii-delay_units+1,np.arange(0,np.shape(validation_data)[1])+np.shape(validation_data)[1]*delay]=validation_data[iii-delay,:]
                    recurrent_validating_labels[iii-delay_units+1]=validation_labels[iii]
                    
                
                ## renaming the training_data and training_labels for the lag ones
                validation_labels=recurrent_validating_labels
                validation_data=recurrent_validation_data
               
                ########################### Lagging ###########################
                
                ## apply the training standardization
                validation_data=Classifier.applyPreprocess(validation_data,mean,std)
                
                # selecting the non-redundant features of training only
                validation_data=validation_data[:,good_features_indexes]  
                
                # predicting samples
                validation_predicted=Classifier.classify(classifier,validation_data)
                
                # firing power
                firing_power_size=sop_period
                threshold=0.70
                    
                predicted_scores=scipy.signal.lfilter(Filter.movingAverageFilter(firing_power_size),
                                                   1,np.array(validation_predicted))
                predicted=np.where(np.array(predicted_scores) >= threshold, 1, 0)
                    
                    
                validation_labels=np.squeeze(np.asarray(validation_labels))
                predicted_scores_2=np.array(predicted_scores)
                validation_2_labels=np.array(validation_labels)
                predicted_labels_2=np.array(predicted)
                
                # modification of the filter to the real pipeline
                # the alarm will only be fired when the last consecutive alarm is finished
                sop_bar=0
                seizure_activated=0
                for index_analysis in range (0,len(predicted_labels_2)):
                    if predicted_labels_2[index_analysis]==1 and seizure_activated==0:
                        seizure_activated=1
                        sop_bar=sop_bar+1
                        
                    elif predicted_labels_2[index_analysis]==1 and seizure_activated==1:
                        sop_bar=sop_bar+1
                        if sop_bar==sop_period:
                            seizure_activated=0
                            sop_bar=0
                            predicted_labels_2[index_analysis]=2
                            if index_analysis+1<len(predicted_labels_2):
                                predicted_labels_2[index_analysis+1]=0
                            
                    elif predicted_labels_2[index_analysis]==0 and seizure_activated==1:
                        sop_bar=0
                        seizure_activated=0
                        predicted_labels_2[index_analysis]=2
                
                #the last label must be an alarm, ISTO TEM QUE SER DISCUTIDO pq
                # nao usei as samples de SPH
                if predicted_labels_2[-1]==1:
                    predicted_labels_2[-1]=2
                
                predicted_labels_2=np.where(predicted_labels_2>1.5,1,0)  
                                
                ## refractory behavior
                sop_bar=sop_period+10
                refractory_activation=0
                for index_analysis in range (0,len(predicted_labels_2)):
                    if predicted_labels_2[index_analysis]==1 and refractory_activation==0:
                       refractory_activation=1
                    elif refractory_activation==1:
                        sop_bar=sop_bar-1
                        predicted_labels_2[index_analysis]=0
                        if sop_bar==0:
                            refractory_activation=0
                            sop_bar=sop_period+10
                
                
                roc_curves.append(Classifier.calculateAUC(validation_2_labels,
                                                              predicted_scores_2))
                    
                confusion_matrix=Classifier.confusionMatrix(validation_2_labels,
                                                                predicted_labels_2)
                    
                sensibility.append(Classifier.sensitivity(confusion_matrix))
                false_positive_rates.append(Classifier.specificity(confusion_matrix))
                false_positive_hour_rates.append(Classifier.falsePositiveRateHour(validation_2_labels,
                                                                predicted_labels_2),sop_period)

                number_of_seizures.append(Filter.calculateNumberOfSeizures(validation_2_labels))
                number_of_triggered_seizures.append(Filter.calculateNumberOfTriggeredSeizures(validation_2_labels,
                                                                                             predicted_labels_2))
                    
                number_of_false_alarms.append(Filter.calculateNumberOfFalseAlarms(validation_2_labels,
                                                                                             predicted_labels_2))
                    
                #ratio_correct_seizures=(Filter.calculateNumberOfTriggeredSeizures(validation_2_labels,predicted_labels_2)/
                                            #Filter.calculateNumberOfSeizures(validation_2_labels))
                                            
                false_alarms_hour=Classifier.falsePositiveRateHour(validation_2_labels,predicted_labels_2,sop_period)
                time_under_false_alarm=Classifier.timeUnderFalseAlarm(validation_2_labels,predicted_labels_2)
                ratio_penalty=false_alarms_hour*(1+time_under_false_alarm)
                ratio_penalties.append(ratio_penalty)
                
                
                fitness.append(Classifier.sensitivity(confusion_matrix)-ratio_penalty)
                
                
        if metrics == "fitness":
            metric=fitness
            
       
        
        
        
        # sensibility and false positive rate per hour
        elif metrics == "sensibility_fpr_hour":
             metric=np.transpose(np.vstack((np.array(patient_number),
                                           np.array(np.sum(number_of_seizures)),
                                           np.array(np.mean(sensibility)),
                                           np.array(np.mean(false_positive_hour_rates)),
                                           np.array(np.sum(number_of_false_alarms)),
                                           np.array(np.sum(number_of_triggered_seizures)),
                                           np.array(np.mean(fitness)),
                                           np.array(np.mean(ratio_penalties)))))
             
 
          
                      
             
        # the mean of the metrics
        if method_metrics == "mean":
            metric=np.mean(metric)
            
        # the metrics for all patients specifically    
        elif method_metrics == "all_patients":
            metric=metric
        
  
        return metric
    
            
                            
    
    def applyFilterToDatabaseWithPast(self,filter,objective,metrics,method_metrics,sop_period):   
        preictal_time, filter_size, filter_components=Filter.decomposeFilter(filter)
        if sop_period==0:
            sop_period=preictal_time
        elif sop_period!=0:
           sop_period=preictal_time    
    
        sop_period=preictal_time            
        for patient in self.patients:            
            
            patient_number=patient.patient_number
            roc_curves=[]
            sensibility=[]
            false_positive_rates=[]
            false_positive_hour_rates=[]
            fitness=[]
            ratio_penalties=[]
            
            number_of_seizures=[]
            number_of_triggered_seizures=[]
            number_of_false_alarms=[]
            total_interictal_period=[]
            total_refractory_lost=[]
            
            
            # getting seizures
            if objective == "validation_data":
                validation_stage="train"
            elif objective == "test_data":
                validation_stage="test"
                
            # get seizure indexes (for training or for testing)
            seizures_indexes=patient.getSeizureIndexesFor(validation_stage)
            
            # iterating the selected seizures
            for index in range(0,len(seizures_indexes)-1):
                                
                #################### training procedure ####################
                past_data=[]
                past_labels=[]
                for past in range(0,seizures_indexes[index]+1):
                    # training seizure
                    training_data=patient.getSeizureIndexData(past,
                                                              filter_size, 
                                                              preictal_time, 
                                                              filter_components,
                                                              validation_stage)
                                     
                    ## Splitting into labels and features
                    training_labels=training_data[:,-1]
                    training_data=training_data[:,:-1]
                    
                    
                    ########################### Lagging ###########################
    
                    # constructing the feature delays
                    delay_units=3
                    # features with lags
                    recurrent_training_data=np.zeros((len(training_labels)-delay_units,
                                                        np.shape(training_data)[1]*delay_units))              
                    # labels with lags
                    recurrent_training_labels=np.zeros(len(training_labels)-delay_units)
                    
                
                    # retrieving the lags
                    for iii in range(delay_units-1,len(training_labels)-1):
                        for delay in range(0,delay_units):
                            recurrent_training_data[iii-delay_units+1,np.arange(0,np.shape(training_data)[1])+np.shape(training_data)[1]*delay]=training_data[iii-delay,:]
                        recurrent_training_labels[iii-delay_units+1]=training_labels[iii]
                        
                    
                    ## renaming the training_data and training_labels for the lag ones
                    training_labels=recurrent_training_labels
                    training_data=recurrent_training_data
                    
                    # if it is the first seizure, that is, 
                    # if past_data and past_labels are empty, the np.arays are created
                    if len(past_data)==0:
                        past_data=training_data
                        past_labels=training_labels
                        
                    # if it is the next seizures, the new data is concatenated to the
                    # old one
                    else:
                        past_data=np.concatenate((past_data,training_data),axis=0)
                        past_labels=np.concatenate((past_labels,training_labels))
                    
                
                    
                
                # the training data is now all past data and past labels
                training_data=past_data
                training_labels=past_labels
                
                
                ########################### Lagging ###########################

                
                #Preprocessing the data            
                training_data,mean,std=Classifier.preProcessTrainingData(training_data)
                
                redundant_features=[]
                #get redundant feature with a threshold higher than 0.95
                for feature_index in range(np.shape(training_data)[1]-1,-1,-1):
                    for feature_index_j in range(np.shape(training_data)[1]-2,0,-1):
                        if feature_index==feature_index_j:
                            continue
                        elif ((feature_index in redundant_features) or 
                              (feature_index_j in redundant_features)):
                            continue
                        else:
                            feature_a=training_data[:,feature_index]
                            feature_b=training_data[:,feature_index_j]
                            if abs(np.corrcoef(feature_a,feature_b)[0][1])>0.95:
                                redundant_features.append(feature_index)
                
                # create a vector with non-redundant features
                good_features_indexes=np.delete(np.arange(0,np.shape(training_data)[1]),
                                                redundant_features)
                # selecting the non-redundant features only
                training_data=training_data[:,good_features_indexes]                        
                
                                
                # First Phase training
                classifier=Classifier.trainClassifier(training_data,training_labels,"over")
                                
                
                #################### validation procedure ####################
                
                # testing seizure
                validation_data=patient.getSeizureIndexData(seizures_indexes[index+1],
                                                          filter_size, 
                                                          preictal_time, 
                                                          filter_components,
                                                          validation_stage)
                 
                
                ## Splitting into labels and features
                validation_labels=validation_data[:,-1]
                validation_data=validation_data[:,:-1]
                
                 ########################### Lagging ###########################

                # features with lags
                recurrent_validation_data=np.zeros((len(validation_labels)-delay_units,
                                                    np.shape(validation_data)[1]*delay_units))              
                # labels with lags
                recurrent_validating_labels=np.zeros(len(validation_labels)-delay_units)
                
                # retrieving the lags
                for iii in range(delay_units-1,len(validation_labels)-1):
                    for delay in range(0,delay_units):
                        recurrent_validation_data[iii-delay_units+1,np.arange(0,np.shape(validation_data)[1])+np.shape(validation_data)[1]*delay]=validation_data[iii-delay,:]
                    recurrent_validating_labels[iii-delay_units+1]=validation_labels[iii]
                    
                
                ## renaming the training_data and training_labels for the lag ones
                validation_labels=recurrent_validating_labels
                validation_data=recurrent_validation_data
               
                ########################### Lagging ###########################
                
                ## apply the training standardization
                validation_data=Classifier.applyPreprocess(validation_data,mean,std)
                
                # selecting the non-redundant features of training only
                validation_data=validation_data[:,good_features_indexes]  
                
                # predicting samples
                validation_predicted=Classifier.classify(classifier,validation_data)
                
                # firing power
                firing_power_size=int(sop_period/Filter.getStepFilterInMinutesInterictal())
                threshold=0.70
                    
                predicted_scores=scipy.signal.lfilter(Filter.movingAverageFilter(firing_power_size),
                                                   1,np.array(validation_predicted))
                predicted=np.where(np.array(predicted_scores) >= threshold, 1, 0)
                    
                    
                validation_labels=np.squeeze(np.asarray(validation_labels))
                predicted_scores_2=np.array(predicted_scores)
                validation_2_labels=np.array(validation_labels)
                predicted_labels_2=np.array(predicted)
                
                if not objective=="validation_data":
                     
#                    predicted_labels_2=np.diff(predicted_labels_2);
#                    predicted_labels_2=np.where(np.array(predicted_labels_2) >= 1, 1, 0);
#                    predicted_labels_2 = np.insert(predicted_labels_2, 0, 0, axis=0)
#                    
                    ## refractory behavior
                    sop_bar=sop_period/Filter.getStepFilterInMinutesInterictal()+10/Filter.getStepFilterInMinutesInterictal()
                    refractory_activation=0
                    for index_analysis in range (0,len(predicted_labels_2)):
                        if predicted_labels_2[index_analysis]==1 and refractory_activation==0:
                           refractory_activation=1
                        elif refractory_activation==1:
                            sop_bar=sop_bar-1
                            predicted_labels_2[index_analysis]=0
                            if sop_bar==0:
                                refractory_activation=0
                                sop_bar=sop_period/Filter.getStepFilterInMinutesInterictal()+10/Filter.getStepFilterInMinutesInterictal()
                              
                           
                
                
                roc_curves.append(Classifier.calculateAUC(validation_2_labels,
                                                              predicted_scores_2))
                    
                confusion_matrix=Classifier.confusionMatrix(validation_2_labels,
                                                                predicted_labels_2)
                    
                sensibility.append(Classifier.sensitivity(confusion_matrix))
                false_positive_rates.append(Classifier.specificity(confusion_matrix))
                
                if not objective=="validation_data":
                    false_positive_hour_rates.append(Classifier.falsePositiveRateHour(validation_2_labels,
                                                                predicted_labels_2,sop_period))
                else:
                    false_positive_hour_rates.append(Classifier.falsePositiveRateHourNoRefractoryPeriod(validation_2_labels,
                                                                predicted_labels_2,sop_period))

                number_of_seizures.append(Filter.calculateNumberOfSeizures(validation_2_labels))
                number_of_triggered_seizures.append(Filter.calculateNumberOfTriggeredSeizures(validation_2_labels,
                                                                                             predicted_labels_2))
                    
                number_of_false_alarms.append(Filter.calculateNumberOfFalseAlarms(validation_2_labels,
                                                                                             predicted_labels_2))
                    
                #ratio_correct_seizures=(Filter.calculateNumberOfTriggeredSeizures(validation_2_labels,predicted_labels_2)/
                                            #Filter.calculateNumberOfSeizures(validation_2_labels))
                                            
                false_alarms_hour=Classifier.falsePositiveRateHourNoRefractoryPeriod(validation_2_labels,predicted_labels_2,sop_period)
                time_under_false_alarm=Classifier.timeUnderFalseAlarm(validation_2_labels,predicted_labels_2)
                ratio_penalty=false_alarms_hour*(1+time_under_false_alarm)
                ratio_penalties.append(ratio_penalty)
                
                # new objective function
                fitness.append(0.5*Classifier.sensitivity(confusion_matrix)+
                               0.5*Filter.calculateNumberOfTriggeredSeizures(validation_2_labels,
                                                                                             predicted_labels_2)
                               -ratio_penalty)
                        
                
                # old objective function
                #fitness.append(Classifier.sensitivity(confusion_matrix)-ratio_penalty)
                
                
                # for the test seizures 
                total_interictal_period.append(Classifier.getInterIctalTotalPeriod(validation_2_labels,
                                                                predicted_labels_2,sop_period))
                total_refractory_lost.append(Classifier.LostRefractoryTime(validation_2_labels,
                                                                predicted_labels_2,sop_period))
                
                
        if metrics == "fitness":
            metric=fitness
            
       
        
        
        
        # sensibility and false positive rate per hour
        elif metrics == "sensibility_fpr_hour":
             metric=np.transpose(np.vstack((np.array(patient_number),
                                           np.array(np.sum(number_of_seizures)),
                                           np.array(np.mean(sensibility)),
                                           # for training
#                                           np.array(np.mean(false_positive_hour_rates)),
                                           # for testing
                                           np.array(np.sum(number_of_false_alarms)/(np.sum(total_interictal_period)-np.sum(total_refractory_lost))),
                                           np.array(np.sum(number_of_false_alarms)),
                                           np.array(np.sum(number_of_triggered_seizures)),
                                           np.array(np.mean(fitness)),
                                           np.array(np.mean(ratio_penalties)),
                                           np.array(np.mean(preictal_time)))))
  
          
                      
             
        # the mean of the metrics
        if method_metrics == "mean":
            metric=np.mean(metric)
            
        # the metrics for all patients specifically    
        elif method_metrics == "all_patients":
            metric=metric
        
  
        return metric
    
    
    
    
    def calculateFilterPredictivePowerWithPast(self,filter,objective,sop_period):   
        preictal_time, filter_size, filter_components=Filter.decomposeFilter(filter)
        if sop_period==0:
            sop_period=preictal_time
        else:
            preictal_time=sop_period
            
        sop_period=sop_period
        preictal_time=sop_period
                    
        for patient in self.patients:            
            
            # constructing the feature delays
            delay_units=3
            features_predictive_power=np.zeros(delay_units*len(filter_components))
            
            
            # getting seizures
            if objective == "validation_data":
                validation_stage="train"
            elif objective == "test_data":
                validation_stage="test"
                
            # get seizure indexes (for training or for testing)
            seizures_indexes=patient.getSeizureIndexesFor(validation_stage)
            
            # iterating the selected seizures
            for index in range(0,len(seizures_indexes)-1):
                                
                #################### training procedure ####################
                past_data=[]
                past_labels=[]
                for past in range(0,seizures_indexes[index]+1):
                    # training seizure
                    training_data=patient.getSeizureIndexData(past,
                                                              filter_size, 
                                                              preictal_time, 
                                                              filter_components,
                                                              validation_stage)
                                     
                    ## Splitting into labels and features
                    training_labels=training_data[:,-1]
                    training_data=training_data[:,:-1]
                    
                    
                    ########################### Lagging ###########################
    
                    # features with lags
                    recurrent_training_data=np.zeros((len(training_labels)-delay_units,
                                                        np.shape(training_data)[1]*delay_units))              
                    # labels with lags
                    recurrent_training_labels=np.zeros(len(training_labels)-delay_units)
                    
                
                    # retrieving the lags
                    for iii in range(delay_units-1,len(training_labels)-1):
                        for delay in range(0,delay_units):
                            recurrent_training_data[iii-delay_units+1,np.arange(0,np.shape(training_data)[1])+np.shape(training_data)[1]*delay]=training_data[iii-delay,:]
                        recurrent_training_labels[iii-delay_units+1]=training_labels[iii]
                        
                    
                    ## renaming the training_data and training_labels for the lag ones
                    training_labels=recurrent_training_labels
                    training_data=recurrent_training_data
                    
                    # if it is the first seizure, that is, 
                    # if past_data and past_labels are empty, the np.arays are created
                    if len(past_data)==0:
                        past_data=training_data
                        past_labels=training_labels
                        
                    # if it is the next seizures, the new data is concatenated to the
                    # old one
                    else:
                        past_data=np.concatenate((past_data,training_data),axis=0)
                        past_labels=np.concatenate((past_labels,training_labels))
                    
                
                    
                
                # the training data is now all past data and past labels
                training_data=past_data
                training_labels=past_labels
                
                
                ########################### Lagging ###########################

                
                #Preprocessing the data            
                training_data,mean,std=Classifier.preProcessTrainingData(training_data)
                
                redundant_features=[]
                #get redundant feature with a threshold higher than 0.95
                for feature_index in range(np.shape(training_data)[1]-1,-1,-1):
                    for feature_index_j in range(np.shape(training_data)[1]-2,0,-1):
                        if feature_index==feature_index_j:
                            continue
                        elif ((feature_index in redundant_features) or 
                              (feature_index_j in redundant_features)):
                            continue
                        else:
                            feature_a=training_data[:,feature_index]
                            feature_b=training_data[:,feature_index_j]
                            if abs(np.corrcoef(feature_a,feature_b)[0][1])>0.95:
                                redundant_features.append(feature_index)
                
                # create a vector with non-redundant features
                good_features_indexes=np.delete(np.arange(0,np.shape(training_data)[1]),
                                                redundant_features)
                # selecting the non-redundant features only
                training_data=training_data[:,good_features_indexes]                        
                
                                
                # First Phase training
                classifier=Classifier.trainClassifier(training_data,training_labels,"over")
                
                # Predictive Power
                for index_predictive in range(0,len(good_features_indexes)):
                    features_predictive_power[good_features_indexes[index_predictive]]=(features_predictive_power[good_features_indexes[index_predictive]]
                    +abs(classifier.coef_[0][index_predictive]))
                                                                    
        
        # merging the lagging features to the mother feature
        predictive_power=np.zeros(len(filter_components))
        for feature_index in range(0,len(filter_components)):
            predictive_power[feature_index]=np.sum(features_predictive_power[feature_index*3:feature_index*3+3])
        
  
        return predictive_power
    
    
    
    
    
    
    def getTrainedClassifierAndSamples(self,filter,objective,sop_period):   
        preictal_time, filter_size, filter_components=Filter.decomposeFilter(filter)
        if sop_period==0:
            sop_period=preictal_time
        else:
            preictal_time=sop_period
            
        sop_period=sop_period
        preictal_time=sop_period
                    
        for patient in self.patients:            
            
            # constructing the feature delays
            delay_units=3
            features_predictive_power=np.zeros(delay_units*len(filter_components))
            
            
            # getting seizures
            if objective == "validation_data":
                validation_stage="train"
            elif objective == "test_data":
                validation_stage="test"
                
            # get seizure indexes (for training or for testing)
            seizures_indexes=patient.getSeizureIndexesFor(validation_stage)
            
            # iterating the selected seizures
            for index in range(0,len(seizures_indexes)-1):
                                
                #################### training procedure ####################
                past_data=[]
                past_labels=[]
                for past in range(0,seizures_indexes[index]+1):
                    # training seizure
                    training_data=patient.getSeizureIndexData(past,
                                                              filter_size, 
                                                              preictal_time, 
                                                              filter_components,
                                                              validation_stage)
                                     
                    ## Splitting into labels and features
                    training_labels=training_data[:,-1]
                    training_data=training_data[:,:-1]
                    
                    
                    ########################### Lagging ###########################
    
                    # features with lags
                    recurrent_training_data=np.zeros((len(training_labels)-delay_units,
                                                        np.shape(training_data)[1]*delay_units))              
                    # labels with lags
                    recurrent_training_labels=np.zeros(len(training_labels)-delay_units)
                    
                
                    # retrieving the lags
                    for iii in range(delay_units-1,len(training_labels)-1):
                        for delay in range(0,delay_units):
                            recurrent_training_data[iii-delay_units+1,np.arange(0,np.shape(training_data)[1])+np.shape(training_data)[1]*delay]=training_data[iii-delay,:]
                        recurrent_training_labels[iii-delay_units+1]=training_labels[iii]
                        
                    
                    ## renaming the training_data and training_labels for the lag ones
                    training_labels=recurrent_training_labels
                    training_data=recurrent_training_data
                    
                    # if it is the first seizure, that is, 
                    # if past_data and past_labels are empty, the np.arays are created
                    if len(past_data)==0:
                        past_data=training_data
                        past_labels=training_labels
                        
                    # if it is the next seizures, the new data is concatenated to the
                    # old one
                    else:
                        past_data=np.concatenate((past_data,training_data),axis=0)
                        past_labels=np.concatenate((past_labels,training_labels))
                    
                
                    
                
                # the training data is now all past data and past labels
                training_data=past_data
                training_labels=past_labels
                
                
                ########################### Lagging ###########################

                
                #Preprocessing the data            
                training_data,mean,std=Classifier.preProcessTrainingData(training_data)
                
                redundant_features=[]
                #get redundant feature with a threshold higher than 0.95
                for feature_index in range(np.shape(training_data)[1]-1,-1,-1):
                    for feature_index_j in range(np.shape(training_data)[1]-2,0,-1):
                        if feature_index==feature_index_j:
                            continue
                        elif ((feature_index in redundant_features) or 
                              (feature_index_j in redundant_features)):
                            continue
                        else:
                            feature_a=training_data[:,feature_index]
                            feature_b=training_data[:,feature_index_j]
                            if abs(np.corrcoef(feature_a,feature_b)[0][1])>0.95:
                                redundant_features.append(feature_index)
                
                # create a vector with non-redundant features
                good_features_indexes=np.delete(np.arange(0,np.shape(training_data)[1]),
                                                redundant_features)
                # selecting the non-redundant features only
                training_data=training_data[:,good_features_indexes]                        
                
                                
                # First Phase training
                classifier=Classifier.trainClassifier(training_data,training_labels,"over")
                
                # Predictive Power
                for index_predictive in range(0,len(good_features_indexes)):
                    features_predictive_power[good_features_indexes[index_predictive]]=(features_predictive_power[good_features_indexes[index_predictive]]
                    +abs(classifier.coef_[0][index_predictive]))
                                                                    
        
        # merging the lagging features to the mother feature
        predictive_power=np.zeros(len(filter_components))
        for feature_index in range(0,len(filter_components)):
            predictive_power[feature_index]=np.sum(features_predictive_power[feature_index*3:feature_index*3+3])
        
  
        return classifier, training_data, training_labels
    
    
    
    
    def surrogateAnalysis(self,filter,objective,metrics,method_metrics,sop_period,number_of_surrogates):   
        preictal_time, filter_size, filter_components=Filter.decomposeFilter(filter)  
        if sop_period==0:
            sop_period=preictal_time
        #else:
           # preictal_time=sop_period
                    
        for patient in self.patients:  
            
            
            patient_number=patient.patient_number
            roc_curves=[]
            sensibility=[]
            false_positive_rates=[]
            false_positive_hour_rates=[]
            fitness=[]
            ratio_penalties=[]
                
            number_of_seizures=[]
            number_of_triggered_seizures=[]
            number_of_false_alarms=[]
               
            # for the test seizures 
            total_interictal_period=[];
            total_refractory_lost=[];
            
            for k_iterations in range(0,number_of_surrogates):
            
#                patient_number=patient.patient_number
#                roc_curves=[]
#                sensibility=[]
#                false_positive_rates=[]
#                false_positive_hour_rates=[]
#                fitness=[]
#                ratio_penalties=[]
#                
#                number_of_seizures=[]
#                number_of_triggered_seizures=[]
#                number_of_false_alarms=[]
#                
#                # for the test seizures 
#                total_interictal_period=[];
#                total_refractory_lost=[];
                
                # getting seizures
                if objective == "validation_data":
                    validation_stage="train"
                elif objective == "test_data":
                    validation_stage="test"
                    
                # get seizure indexes (for training or for testing)
                seizures_indexes=patient.getSeizureIndexesFor(validation_stage)
                
                # iterating the selected seizures
                for index in range(0,len(seizures_indexes)-1):
                                    
                    #################### training procedure ####################
                    past_data=[]
                    past_labels=[]
                    for past in range(0,seizures_indexes[index]+1):
                        # training seizure
                        training_data=patient.getSeizureIndexData(past,
                                                                  filter_size, 
                                                                  preictal_time, 
                                                                  filter_components,
                                                                  validation_stage)
                                         
                        ## Splitting into labels and features
                        training_labels=training_data[:,-1]
                        training_data=training_data[:,:-1]
                        
                        
                        ########################### Lagging ###########################
        
                        # constructing the feature delays
                        delay_units=3
                        # features with lags
                        recurrent_training_data=np.zeros((len(training_labels)-delay_units,
                                                            np.shape(training_data)[1]*delay_units))              
                        # labels with lags
                        recurrent_training_labels=np.zeros(len(training_labels)-delay_units)
                        
                    
                        # retrieving the lags
                        for iii in range(delay_units-1,len(training_labels)-1):
                            for delay in range(0,delay_units):
                                recurrent_training_data[iii-delay_units+1,np.arange(0,np.shape(training_data)[1])+np.shape(training_data)[1]*delay]=training_data[iii-delay,:]
                            recurrent_training_labels[iii-delay_units+1]=training_labels[iii]
                            
                        
                        ## renaming the training_data and training_labels for the lag ones
                        training_labels=recurrent_training_labels
                        training_data=recurrent_training_data
                        
                        # if it is the first seizure, that is, 
                        # if past_data and past_labels are empty, the np.arays are created
                        if len(past_data)==0:
                            past_data=training_data
                            past_labels=training_labels
                            
                        # if it is the next seizures, the new data is concatenated to the
                        # old one
                        else:
                            past_data=np.concatenate((past_data,training_data),axis=0)
                            past_labels=np.concatenate((past_labels,training_labels))
                        
                    
                        
                    
                    # the training data is now all past data and past labels
                    training_data=past_data
                    training_labels=past_labels
                    
                    
                    ########################### Lagging ###########################
    
                    
                    #Preprocessing the data            
                    training_data,mean,std=Classifier.preProcessTrainingData(training_data)
                    
                    redundant_features=[]
                    #get redundant feature with a threshold higher than 0.95
                    for feature_index in range(np.shape(training_data)[1]-1,-1,-1):
                        for feature_index_j in range(np.shape(training_data)[1]-2,0,-1):
                            if feature_index==feature_index_j:
                                continue
                            elif ((feature_index in redundant_features) or 
                                  (feature_index_j in redundant_features)):
                                continue
                            else:
                                feature_a=training_data[:,feature_index]
                                feature_b=training_data[:,feature_index_j]
                                if abs(np.corrcoef(feature_a,feature_b)[0][1])>0.95:
                                    redundant_features.append(feature_index)
                    
                    # create a vector with non-redundant features
                    good_features_indexes=np.delete(np.arange(0,np.shape(training_data)[1]),
                                                    redundant_features)
                    # selecting the non-redundant features only
                    training_data=training_data[:,good_features_indexes]                        
                    
                                    
                    # First Phase training
                    classifier=Classifier.trainClassifier(training_data,training_labels,"over")
                                    
                    
                    #################### validation procedure ####################
                    
                    # testing seizure
                    validation_data=patient.getSeizureIndexData(seizures_indexes[index+1],
                                                              filter_size, 
                                                              preictal_time, 
                                                              filter_components,
                                                              validation_stage)
                     
                    
                    ## Splitting into labels and features
                    validation_labels=validation_data[:,-1]
                    validation_data=validation_data[:,:-1]
                    
                     ########################### Lagging ###########################
    
                    # features with lags
                    recurrent_validation_data=np.zeros((len(validation_labels)-delay_units,
                                                        np.shape(validation_data)[1]*delay_units))              
                    # labels with lags
                    recurrent_validating_labels=np.zeros(len(validation_labels)-delay_units)
                    
                    # retrieving the lags
                    for iii in range(delay_units-1,len(validation_labels)-1):
                        for delay in range(0,delay_units):
                            recurrent_validation_data[iii-delay_units+1,np.arange(0,np.shape(validation_data)[1])+np.shape(validation_data)[1]*delay]=validation_data[iii-delay,:]
                        recurrent_validating_labels[iii-delay_units+1]=validation_labels[iii]
                        
                    
                    ## renaming the training_data and training_labels for the lag ones
                    validation_labels=recurrent_validating_labels
                    validation_data=recurrent_validation_data
                   
                    ########################### Lagging ###########################
                    
                    ## apply the training standardization
                    validation_data=Classifier.applyPreprocess(validation_data,mean,std)
                    
                    # selecting the non-redundant features of training only
                    validation_data=validation_data[:,good_features_indexes]  
                    
                    # predicting samples
                    validation_predicted=Classifier.classify(classifier,validation_data)
                    
                    # firing power
                    firing_power_size=sop_period
                    threshold=0.70
                        
                    predicted_scores=scipy.signal.lfilter(Filter.movingAverageFilter(firing_power_size),
                                                       1,np.array(validation_predicted))
                    predicted=np.where(np.array(predicted_scores) >= threshold, 1, 0)
                        
                        
                    validation_labels=np.squeeze(np.asarray(validation_labels))
                    predicted_scores_2=np.array(predicted_scores)
                    validation_2_labels=np.array(validation_labels)
                    predicted_labels_2=np.array(predicted)
                    
                                    
                    ## refractory behavior
                    sop_bar=sop_period/Filter.getStepFilterInMinutesInterictal()+10/Filter.getStepFilterInMinutesInterictal()
                    refractory_activation=0
                    for index_analysis in range (0,len(predicted_labels_2)):
                        if predicted_labels_2[index_analysis]==1 and refractory_activation==0:
                           refractory_activation=1
                        elif refractory_activation==1:
                            sop_bar=sop_bar-1
                            predicted_labels_2[index_analysis]=0
                            if sop_bar==0:
                                refractory_activation=0
                                sop_bar=sop_period/Filter.getStepFilterInMinutesInterictal()+10/Filter.getStepFilterInMinutesInterictal()
                           
                    
                    validation_2_labels=Database.applySurrogateToLabels(validation_2_labels)
                    roc_curves.append(Classifier.calculateAUC(validation_2_labels,
                                                                  predicted_scores_2))
                        
                    confusion_matrix=Classifier.confusionMatrix(validation_2_labels,
                                                                    predicted_labels_2)
                        
                    sensibility.append(Classifier.sensitivity(confusion_matrix))
                    false_positive_rates.append(Classifier.specificity(confusion_matrix))
                    false_positive_hour_rates.append(Classifier.falsePosititiveRateHourSurrogate(validation_2_labels,
                                                                    predicted_labels_2,sop_period))
                    
    
                    number_of_seizures.append(Filter.calculateNumberOfSeizures(validation_2_labels))
                    number_of_triggered_seizures.append(Filter.calculateNumberOfTriggeredSeizuresSurrogate(validation_2_labels,
                                                                                                 predicted_labels_2))
                    
                    number_of_false_alarms.append(Filter.calculateNumberOfFalseAlarmsSurrogate(validation_2_labels,
                                                                                                 predicted_labels_2))
                        
                    false_alarms_hour=Classifier.falsePosititiveRateHourSurrogate(validation_2_labels,predicted_labels_2,sop_period)
                    time_under_false_alarm=Classifier.timeUnderFalseAlarm(validation_2_labels,predicted_labels_2)
                    ratio_penalty=false_alarms_hour*(1+time_under_false_alarm)
                    ratio_penalties.append(ratio_penalty)
                    
                    fitness.append(Classifier.sensitivity(confusion_matrix)-ratio_penalty)
                    
                    # for the test seizures 
                    total_interictal_period.append(Classifier.getInterIctalTotalPeriod(validation_2_labels,
                                                                predicted_labels_2,sop_period))
                    total_refractory_lost.append(Classifier.LostRefractoryTimeSurrogate(validation_2_labels,
                                                                predicted_labels_2,sop_period))
                
                
        if metrics == "fitness":
            metric=fitness
            
       # sensibility and false positive rate per hour
        elif metrics == "sensibility_fpr_hour":
             metric=np.transpose(np.vstack((np.array(patient_number),
                                           np.array(np.sum(number_of_seizures)),
                                           np.array(np.mean(sensibility)),
                                           # for training
#                                           np.array(np.mean(false_positive_hour_rates)),
                                           # for testing
                                           np.array(np.sum(number_of_false_alarms)/(np.sum(total_interictal_period)-np.sum(total_refractory_lost))),
                                           np.array(np.sum(number_of_false_alarms)),
                                           np.array(np.sum(number_of_triggered_seizures)),
                                           np.array(np.mean(fitness)),
                                           np.array(np.mean(ratio_penalties)),
                                           np.array(np.mean(preictal_time)),
                                           np.array(np.std(number_of_triggered_seizures)))))
        
  
          
                      
             
        # the mean of the metrics
        if method_metrics == "mean":
            metric=np.mean(metric)
            
        # the metrics for all patients specifically    
        elif method_metrics == "all_patients":
            metric=metric
        
  
        return metric
    
    
    def applySurrogateToLabels(labels):
        preictal_samples_number=np.sum(labels)
        surrogate_data=np.zeros(len(labels))
        seizure_begin=np.random.choice(np.arange(1,len(labels)-preictal_samples_number,1))
        for i in range(0,int(preictal_samples_number)):
            surrogate_data[int(i+seizure_begin)]=1
        
        return surrogate_data
    
    
  
        
    
   
  
        

        
    
          
 