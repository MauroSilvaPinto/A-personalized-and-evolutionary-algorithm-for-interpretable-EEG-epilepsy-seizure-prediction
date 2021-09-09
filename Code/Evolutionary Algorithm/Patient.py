"""
Patient class.

Each patient is constituted by an ID and by seizures.
These "seizures" are the data from chronologically first-level features
extracted in windows of non-overlapping 5s, which concern to data before
each seizure.

These features, for each seizure, are called: 
    preictal_pat_[patient_id]_[seizure_number].mat
    
As these were extracted with Matlab.

"""

import os
import scipy.io as sio
import numpy as np
from Filter import Filter

class Patient:
    
    # initializes the patient with the provided directory and id number
    # loads the respective seizures id numbers
    # loads the respective seizures data
    def __init__(self,number,path):
        self.directory=path
        self.patient_number=number
        
        self.seizures_list=[]
        self.loadSeizuresList()
        self.sortSeizuresList()
        
        self.seizures_data=[]
        self.loadAllSeizuresData()
        
    
    # loads the list of seizures id numbers
    def loadSeizuresList(self):
        filenames=self.getFilenames()
        for filename in filenames:
            if (self.isSeizureFromPatient(filename) and
                not "_preictal" in filename):
                    self.seizures_list.append(int(filename.split('_')[3].split('.')[0]))   
    
    # loads all the existing seizures from the patient    
    def loadAllSeizuresData(self):
        for seizure in self.seizures_list:
            filename = "preictal_pat_" + str(self.patient_number) + "_" + str(seizure) + ".mat"
            self.loadSeizureData(filename)
            
    
    # loads the seizure data from the provided .mat file
    def loadSeizureData(self,filename):
         self.seizures_data.append(sio.loadmat(filename)['seizure_data'])
    
    
    # sorts the seizures id's by its chronological order
    def sortSeizuresList(self):
        self.seizures_list.sort()
         
    # retrieves the filenames of the directory provided to get the patient data    
    def getFilenames(self):
        os.chdir(self.directory)
        filenames=os.listdir()    
        return filenames
    
    # confirms if a given filename contains seizure data
    def isFilenameSeizureData(filename):
        return ("preictal" in filename)
    
    # confirms if a given filename contains seizure data and it if belongs
    # to the provided patient
    def isSeizureFromPatient(self, filename):
        return Patient.isFilenameSeizureData(filename) and (str(self.patient_number) in filename)  
        
    
    # get the id number of the patient    
    def getNumber(self):
        return self.patient_number
    
    # print the seizures id numbers of the patient
    def printSeizures(self):
        print(str(len(self.seizures_list)) + " Seizures: " + str(self.seizures_list) + "\n")
      
    # get the number of seizures of the patient
    def getNumberOfSeizures(self):
        return (len(self.seizures_list))
    
    # retrieves the duration of all seizures as a list
    def getSeizuresDurationInSamples(self):
        sample_duration=[]
        for seizure in self.seizures_data:
            sample_duration.append(Patient.getSeizureDuration(seizure))
        return sample_duration
    
    def getSeizureDuration(seizure):
        return seizure.shape[1]
    
    # returns a boolean if the duration of a seizure in samples is under or
    # over the provided time in minutes
    def isSeizureTooSmall(seizure, minutes):
        return (seizure.shape[1]*5/60)<minutes
    
    
    # eliminates seizures from the patient with less than the provided minutes
    def eliminateSeizuresWithLessThan(self,minutes):
        for i in range(len(self.seizures_data)-1,-1,-1):
            if Patient.isSeizureTooSmall(self.seizures_data[i],minutes):
                self.seizures_data.pop(i)
                self.seizures_list.pop(i)
    
    # checks if the classification phase is regarding training or not
    def isTrainingPhase(phase):
        return phase=="train"
    
    # checks if the classification phase is regarding validation or not
    def isValidatingPhase(phase):
        return phase=="validate"
    
    # checks if the classification phase is regarding testing or not
    def isTestPhase(phase):
        return phase=="test"
    
    # retrieves the seizures concerning the phase in question:
    #   training, validation or testing
    # then, for each seizure, extracts the features by applying the filter
    # afterwards, the data for all seizures is concatenated into a numpy array
    #
    # NOT USED ANYMORE I GUESS
    def getData(self,filter_size, preictal_time, filter_components,phase):
        if Patient.isTrainingPhase(phase):
            seizures=self.getTrainingSeizures()        
        elif Patient.isValidatingPhase(phase):
            seizures=self.getValidatingSeizures()
        elif Patient.isTestPhase(phase):
            seizures=self.getTestingSeizures()
        
        data=Patient.getFilterExtractedData(filter_size,preictal_time,
                                            filter_components,seizures,phase)
        data=Patient.assembleAllLists(data)
        return np.matrix(data)
    
    def getSeizureIndexesFor(self,phase):
        if Patient.isTrainingPhase(phase):
            return self.getTrainingSeizuresIndexes(Patient.getPartitionMethod())
        elif Patient.isTestPhase(phase): 
            return self.getTestingSeizuresIndexes(Patient.getPartitionMethod())
        
   
    
    def getSeizureIndexData(self,index,filter_size, preictal_time, filter_components,phase):
        seizure=self.seizures_data[index]
        return np.array(Patient.getSeizureFeatures(seizure,filter_size,preictal_time,
                                                        filter_components,phase))

    # assembles all lists into the same list, that is, the data inserted is a list
    # with features extracted for each seizure. the data is a list of lists, where
    # each element of the list is a list of features regarding each seazure
    # this way, this function removes the separation by seizures and all the data
    # is concatenated into the same list    
    def assembleAllLists(data):
       return [val for sublist in data for val in sublist]
    
    # chooses the partition method
    #   60% for validating, 40% for testing
    def getPartitionMethod():
        return [0.6, 0.4]
    
    # retrieves the index of the seizures of a patient designed for training
    def getTrainingSeizuresIndexes(self,partition_method):
        return np.arange(round(partition_method[0]*len(self.seizures_data)))
    
    # retrieves the index of the seizures of a patient designed for validating
    #
    # currently not used
    def getValidatingSeizuresIndexes(self,partition_method):
        validating_indexes=np.arange(round(partition_method[1]*len(self.seizures_data)))
        return np.add(validating_indexes,round(partition_method[0]*len(self.seizures_data)))
    
    # retrieves the index of the seizures of a patient designed for testing
    def getTestingSeizuresIndexes(self,partition_method):
        training_indexes=self.getTrainingSeizuresIndexes(Patient.getPartitionMethod())
        return (np.delete(np.arange(0,len(self.seizures_data)),training_indexes[0:-1]))
    
    # retrieves the seizures data of a patient designed for validation    
    def getValidatingSeizures(self):
        indexes=self.getValidatingSeizuresIndexes(Patient.getPartitionMethod())
        return [self.seizures_data[i] for i in indexes]
        
    # retrieves the seizures data of a patient designed for training  
    def getTrainingSeizures(self):
        indexes=self.getTrainingSeizuresIndexes(Patient.getPartitionMethod())
        return [self.seizures_data[i] for i in indexes]
    
    # retrieves the seizures data of a patient designed for testing  
    def getTestingSeizures(self):
        indexes=self.getTestingSeizuresIndexes(Patient.getPartitionMethod())
        return [self.seizures_data[i] for i in indexes]
   
    # converts from minutes to windows of five seconds
    def fromMinutesTo5secWindows(size):
        return size*12
   
     # for each seizure, all features are extracted with the filter for all
     # the time series signal for a seizure
     # 
     # note: in training, the interictal feature extraction part is performed 
     #       with a step and the pre-ictal one with a smaller step in order to 
     #       obtain more data for training, since it is the less representative
     #        class
     # note: in validation and testing, both interictal and preictal feature
     #       extraction are performed with the same step
    def getSeizureFeatures(seizure,filter_size,pre_ictal,filter_components,phase):
        seizure_size=Patient.getSeizureDuration(seizure)
        filter_size=Patient.fromMinutesTo5secWindows(filter_size)
        pre_ictal=Patient.fromMinutesTo5secWindows(pre_ictal)
        
        step_filter_interictal=Filter.getStepFilterInMinutesInterictal()
        step_filter_interictal=Patient.fromMinutesTo5secWindows(step_filter_interictal)
                
        if Patient.isTrainingPhase(phase):
                step_filter_preictal=Filter.getStepFilterInMinutesPreictal()
                step_filter_preictal=Patient.fromMinutesTo5secWindows(step_filter_preictal)
        
        elif Patient.isValidatingPhase(phase):
                step_filter_preictal=Filter.getStepFilterInMinutesInterictal()
                step_filter_preictal=Patient.fromMinutesTo5secWindows(step_filter_preictal)  
        elif Patient.isTestPhase(phase):
                step_filter_preictal=Filter.getStepFilterInMinutesInterictal()
                step_filter_preictal=Patient.fromMinutesTo5secWindows(step_filter_preictal)
        
        index_beginning_filter=0
        index_ending_filter=seizure_size-filter_size
        
        
        filter_steps_interictal=np.arange(index_beginning_filter,seizure_size-pre_ictal-filter_size,step_filter_interictal)
        filter_steps_pre_ictal=np.arange(seizure_size-pre_ictal-filter_size,index_ending_filter,step_filter_preictal)
        
        
        step_filter=np.append(filter_steps_interictal,filter_steps_pre_ictal)
        features=[]
        for i in step_filter:
            i=int(i)
            row=[]
            for component in filter_components:
                index_preprocessing=int(component.split("__")[0])
                seizure_portion=seizure[index_preprocessing][i:i+filter_size]
                row.append(Filter.calculateFeature(seizure_portion,component))
            
            row.append(Filter.calculateLabel(i,seizure_size,filter_size,pre_ictal))
            features.append(row)
       
        return features
        
    # gets the filter extracted data, that is:
    #   iterates all selected seizures
    #   in each one, the features are extracted with the used filter
    def getFilterExtractedData(filter_size,preictal_time,filter_components,
                               seizures, phase):
        data_seizures =[]
        for seizure in seizures:
            seizure_features=Patient.getSeizureFeatures(seizure,filter_size,
                                                        preictal_time,
                                                        filter_components,
                                                        phase)
            data_seizures.append(seizure_features)
                   
        return data_seizures     
 
           
        
        
        
    
    
        
               
                