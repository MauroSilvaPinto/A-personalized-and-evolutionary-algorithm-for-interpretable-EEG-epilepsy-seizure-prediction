''' 
Code with functions that are necessary for the phenotype study.
This code should not be executed, and to work, should be placed in
Evolutionary Algorithms folder

'''


import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from Filter import Filter
import networkx as nx
from Feature import Feature

class Analyze:
    

    def plotPatientsData(data,metric,patient_list):
        fig, ax = plt.subplots()
        ax.plot(np.array(data),'ro')
       
        ax.set(xlabel='patients', ylabel=metric,
               title=(metric+" values for All Patients"))
        ax.grid()

        
        plt.xticks((np.arange(len(patient_list))), (patient_list))
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    def plotPatientsDataValidationTest(validation,test,metric,patient_list):
        fig, ax = plt.subplots()
        ax.plot(np.array(validation),'ro',
                np.array(test),'g^')
       
        ax.set(xlabel='patients', ylabel=metric,
               title=(metric+" values for All Patients"))
        ax.grid()
        plt.legend(('Validation', 'Test'),
           loc='lower right')
        
        plt.xticks((np.arange(len(patient_list))), (patient_list))
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    def plotPatientsSeizuresOccurences(number_of_seizures, triggered,
                                       false_alarms, patient_list):
        fig, ax = plt.subplots()
        ax.plot(np.array(number_of_seizures),'k*',
                np.array(triggered),'g^',
                np.array(false_alarms),'ro')
       
        ax.set(xlabel='patients', ylabel="Ocurrences",
               title=("Seizure Situations for All Patients"))
        ax.grid()
        plt.legend(('Number of Seizures', 'Correctly Triggered','False Alarms'),
           loc='upper right')
        
        plt.xticks((np.arange(len(patient_list))), (patient_list))
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    def boxplotPatientsData(data,metric,patient_list):
        fig, ax = plt.subplots()
        ax.boxplot(np.transpose(np.array(data)),'ro')
       
        ax.set(xlabel='patients', ylabel=metric,
               title=(metric+" values for All Patients"))
        ax.grid()

        
        plt.xticks((np.arange(1,len(patient_list)+1)), (patient_list))
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    def plotBoxplotGenerations(data_generations,patient):
        new_data_generations=data_generations[:,np.arange(0,data_generations.shape[1],10)]
        fig, ax = plt.subplots()
        ax.boxplot(np.array(new_data_generations),'ro')
       
        ax.set(xlabel='Generations', ylabel="fitness",
               title=(patient+" Population Fitness Distribution Throughout Evolution"))
        ax.grid()

        evolution=np.arange(0,data_generations.shape[1])
        evolution=evolution[np.arange(0,data_generations.shape[1],10)]

        plt.xticks(np.arange(0,len(evolution)), evolution,rotation=45)
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    def calculateAverageDistanceBetweenFiltersData(filters_1, filters_2):
        average_distance_filter=[]
        for i in range(0,len(filters_1)):
            for j in range(0,len(filters_2)):
                average_distance_filter.append(Filter.calculateDistanceBetweenFilters(filters_1[i],filters_2[j]))
                
        average_distance_filter=np.array(average_distance_filter)
        
        return np.mean(average_distance_filter)
    
    
    def calculateDistancesBetweenFiltersData(filters_1, filters_2):
        average_distance_filter=[]
        indexes_i=[]
        indexes_j=[]
        for i in range(0,len(filters_1)):
            for j in range(0,len(filters_2)):
                average_distance_filter.append(Filter.calculateDistanceBetweenFilters(filters_1[i],filters_2[j]))
                indexes_i.append(i)
                indexes_j.append(j)
                
        average_distance_filter=np.array(average_distance_filter)
        indexes_i=np.array(indexes_i)
        indexes_j=np.array(indexes_j)
        return (average_distance_filter),indexes_i,indexes_j
    
    
    def calculatePredictivePowerElectrode(filters,weights):
        brain_graph=Feature.getElectrodesGraph()
        electrodes_list=list(brain_graph.nodes)
        predictive_power=np.zeros([1,len(electrodes_list)])[0]
        presence=np.zeros([1,len(electrodes_list)])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(electrodes_list)])[0]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                index_electrode=electrodes_list.index(electrode)
                presence_i[index_electrode]=presence_i[index_electrode]+1
                predictive_power[index_electrode]=predictive_power[index_electrode]+abs(weights[i][j-2])
            
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return electrodes_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    def calculatePredictivePowerLobe(filters,weights):
        predictive_power=np.zeros([1,6])[0] # 6 lobes: central, frontal, pre-frontal, occipital, parietal, temporal
        presence=np.zeros([1,6])[0]
        lobes_list=["central", "frontal", "pre-frontal", "occipital", "parietal", "temporal"]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,6])[0]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                lobe=Analyze.getLobe(electrode)
                index_lobe=lobes_list.index(lobe)
                presence_i[index_lobe]=presence_i[index_lobe]+1
                predictive_power[index_lobe]=predictive_power[index_lobe]+abs(weights[i][j-2])
            
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return lobes_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    def calculatePredictivePowerHemisphere(filters,weights):
        predictive_power=np.zeros([1,3])[0] # 6 lobes: left, central, right
        presence=np.zeros([1,3])[0]
        hemisphere_list=["left", "central", "right"]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,3])[0]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                hemisphere=Analyze.getHemisphere(electrode)
                index_hemisphere=hemisphere_list.index(hemisphere)
                presence_i[index_hemisphere]=presence_i[index_hemisphere]+1
                predictive_power[index_hemisphere]=predictive_power[index_hemisphere]+abs(weights[i][j-2])
            
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return hemisphere_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    def getHemisphere(electrode):
        if Analyze.isElectrodeFromCentralHemisphere(electrode):
            return "central"
        elif Analyze.isElectrodeFromLeftHemisphere(electrode):
            return "left"
        elif Analyze.isElectrodeFromRightHemisphere(electrode):
            return "right"    
    
    def isElectrodeFromCentralHemisphere(electrode):
        return electrode in ["CZ","FZ","PZ"]
    
    def isElectrodeFromRightHemisphere(electrode):
        return electrode in ["C4","F4","F8","FP2","O2","P4","T4","T6"]

    def isElectrodeFromLeftHemisphere(electrode):
        return electrode in ["C3","F3","F7","FP1","O1","P3","T3","T5"]
    
    def getLobe(electrode):
        if Analyze.isElectrodeFromCentralLobe(electrode):
            return "central"
        elif Analyze.isElectrodeFromFrontalLobe(electrode):
            return "frontal"
        elif Analyze.isElectrodeFromPreFrontalLobe(electrode):
            return "pre-frontal"
        elif Analyze.isElectrodeFromOccipitalLobe(electrode):
            return "occipital"
        elif Analyze.isElectrodeFromParietalLobe(electrode):
            return "parietal"
        elif Analyze.isElectrodeFromTemporalLobe(electrode):
            return "temporal"
        
    def isElectrodeFromCentralLobe(electrode):
        return electrode in ["C3","C4","CZ"]
    
    def isElectrodeFromFrontalLobe(electrode):
        return electrode in ["F3","F4","F7","F8","FZ"]

    def isElectrodeFromPreFrontalLobe(electrode):
        return electrode in ["FP1","FP2"]

    def isElectrodeFromOccipitalLobe(electrode):    
        return electrode in ["O1","O2"]
    
    def isElectrodeFromParietalLobe(electrode):
        return electrode in ["P3","P4","PZ"]
    
    def isElectrodeFromTemporalLobe(electrode):
        return electrode in ["T3","T4","T5","T6"]
    
    def calculatePredictivePowerCharacteristic(filters,weights):
        characteristics_list=[Feature.getCharacteristicsList()+ Feature.getWavesList()][0]
        predictive_power=np.zeros([1,len(characteristics_list)])[0]
        presence=np.zeros([1,len(characteristics_list)])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(characteristics_list)])[0]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[0].split('_',1)[-1]
                index_characteristic=characteristics_list.index(characteristic)
                presence_i[index_characteristic]=presence_i[index_characteristic]+1
                predictive_power[index_characteristic]=predictive_power[index_characteristic]+abs(weights[i][j-2])
        
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return characteristics_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    def calculatePredictivePowerCharacteristicType(filters,weights):
        characteristics_types=["non-wave", "wave"]
        predictive_power=np.zeros([1,2])[0]# non-wave, wave
        presence=np.zeros([1,2])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(characteristics_types)])[0]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[0].split('_',1)[-1]
                characteristic_type=Analyze.getCharacteristicType(characteristic)
                
                
                index_characteristic_type=characteristics_types.index(characteristic_type)
                presence_i[index_characteristic_type]=presence_i[index_characteristic_type]+1
                predictive_power[index_characteristic_type]=predictive_power[index_characteristic_type]+abs(weights[i][j-2])
        
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return characteristics_types, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    def calculatePredictivePowerWaveType(filters,weights):
        characteristics_types=["non-gamma", "gamma"]
        predictive_power=np.zeros([1,2])[0]# non-wave, wave
        presence=np.zeros([1,2])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(characteristics_types)])[0]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[0].split('_',1)[-1]
                if Analyze.getCharacteristicType(characteristic)=="wave":
                    characteristic_type=Analyze.getCharacteristicWaveType(characteristic)
                    index_characteristic_type=characteristics_types.index(characteristic_type)
                    presence_i[index_characteristic_type]=presence_i[index_characteristic_type]+1
                    predictive_power[index_characteristic_type]=predictive_power[index_characteristic_type]+abs(weights[i][j-2])
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return characteristics_types, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    def getCharacteristicWaveType(characteristic):
        if Analyze.isCharacteristicWaveGamma(characteristic):
            return "gamma"
        else:
            return "non-gamma"
        
    def isCharacteristicWaveGamma(characteristic):
        return "gamma" in characteristic
    
    def getCharacteristicType(characteristic):
        if Analyze.isCharacteristicNonWave(characteristic):
            return "non-wave"
        elif Analyze.isCharacteristicWave(characteristic):
            return "wave"
        
    def isCharacteristicNonWave(characteristic):
        return characteristic in ["mean_freq","band_power","medium_intensity",
                                  "medium_intensity_unormalized","variance"]
        
    def isCharacteristicWave(characteristic):
        return characteristic in ["delta","theta","alpha","beta","gamma_1",
                                  "gamma_2","gamma_3"]
        
        
    
    def calculatePredictivePowerOperators(filters,weights):
        operators_list=Feature.getMathematicalOperatorsList()
        predictive_power=np.zeros([1,len(operators_list)])[0]
        presence=np.zeros([1,len(operators_list)])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(operators_list)])[0]
            for j in range(2,len(filters[i])):
                operator=filters[i][j].split('__')[1]
                index_operator=operators_list.index(operator)
                presence_i[index_operator]=presence_i[index_operator]+1
                predictive_power[index_operator]=predictive_power[index_operator]+abs(weights[i][j-2])
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return operators_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    def calculatePredictivePowerWindowLength(filters,weights):
        windows_list=Feature.getWindowLengthRange()
        predictive_power=np.zeros([1,len(windows_list)])[0]
        presence=np.zeros([1,len(windows_list)])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(windows_list)])[0]
            for j in range(2,len(filters[i])):
                window=filters[i][j].split('__')[-1]
                index_length=np.where(windows_list==int(window))[0][0]
                presence_i[index_length]=presence_i[index_length]+1
                predictive_power[index_length]=predictive_power[index_length]+abs(weights[i][j-2])
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return windows_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
    
    def calculateDifferentWindowsPresence(filters,weights):
        windows_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            windows_i=[]
            for j in range(2,len(filters[i])):
                window=filters[i][j].split('__')[-1]
                windows_i.append(int(window))
            
            window=len(np.unique(windows_i))
            index_windows=np.where(windows_quantity_list==int(window))[0][0]
            presence[index_windows]=presence[index_windows]+1
        
        return windows_quantity_list, presence/len(filters)
    
    
    def calculateDifferentCharacteristicsPresence(filters,weights):
        characteristic_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            characteristics_i=[]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[0].split('_',1)[1]
                characteristics_i.append(characteristic)
            
            characteristic=len(np.unique(characteristics_i))
            index_characteristic=np.where(characteristic_quantity_list==int(characteristic))[0][0]
            presence[index_characteristic]=presence[index_characteristic]+1
        
        return characteristic_quantity_list, presence/len(filters)
    
    def calculateDifferentCharacteristicTypesPresence(filters,weights):
        characteristic_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            characteristics_i=[]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[0].split('_',1)[1]
                characteristic=Analyze.getCharacteristicType(characteristic)
                characteristics_i.append(characteristic)
            
            characteristic=len(np.unique(characteristics_i))
            index_characteristic=np.where(characteristic_quantity_list==int(characteristic))[0][0]
            presence[index_characteristic]=presence[index_characteristic]+1
        
        return characteristic_quantity_list, presence/len(filters)
    
    
    def calculateDifferentOperatorsPresence(filters,weights):
        characteristic_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            characteristics_i=[]
            for j in range(2,len(filters[i])):
                characteristic=filters[i][j].split('__')[1]
                characteristics_i.append(characteristic)
            
            characteristic=len(np.unique(characteristics_i))
            index_characteristic=np.where(characteristic_quantity_list==int(characteristic))[0][0]
            presence[index_characteristic]=presence[index_characteristic]+1
        
        return characteristic_quantity_list, presence/len(filters)
    
    
    
    def calculateDifferentChronologyPresence(filters,weights):
        events_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            events_i=[]
            for j in range(2,len(filters[i])):
                event=filters[i][j].split('__')[-2]
                events_i.append(int(event))
            
            window=len(np.unique(events_i))
            index_events=np.where(events_quantity_list==int(window))[0][0]
            presence[index_events]=presence[index_events]+1
        
        return events_quantity_list, presence/len(filters)
    
    
    def calculateDifferentElectrodesPresence(filters,weights):
        electrodes_quantity_list=np.linspace(1,len(filters[0])-2,len(filters[0])-2,
                                          dtype=int)
        brain_graph=Feature.getElectrodesGraph()
        electrodes_list=list(brain_graph.nodes)
        presence=np.zeros([1,len(filters[0])-2])[0]
        
        for i in range(0,len(filters)):
            electrodes_i=[]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                index_electrode=electrodes_list.index(electrode)
                electrodes_i.append(int(index_electrode))

            electrodes=len(np.unique(electrodes_i))
            index_electrodes=np.where(electrodes_quantity_list==int(electrodes))[0][0]
            presence[index_electrodes]=presence[index_electrodes]+1
        
        return electrodes_quantity_list, presence/len(filters)
    
    
    
    def calculateDifferentLobesPresence(filters,weights):
        lobes_quantity_list=np.linspace(1,6,6,dtype=int)
        lobes_list=["central", "frontal", "pre-frontal", "occipital", "parietal", "temporal"]
        presence=np.zeros([1,6])[0]
        
        for i in range(0,len(filters)):
            lobes_i=[]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                lobe=Analyze.getLobe(electrode)
                index_lobes=lobes_list.index(lobe)
                lobes_i.append(int(index_lobes))
            
            lobes=len(np.unique(lobes_i))
            index_lobes=np.where(lobes_quantity_list==int(lobes))[0][0]
            presence[index_lobes]=presence[index_lobes]+1
        
        return lobes_quantity_list, presence/len(filters)
    
    
    def calculateDifferentHemispheresPresence(filters,weights):
        hemispheres_quantity_list=np.linspace(1,3,3,dtype=int)
        hemisphere_list=["left", "central", "right"]
        presence=np.zeros([1,3])[0]
        
        for i in range(0,len(filters)):
            hemispheres_i=[]
            for j in range(2,len(filters[i])):
                electrode=filters[i][j].split('_')[0]
                hemisphere=Analyze.getHemisphere(electrode)
                index_hemisphere=hemisphere_list.index(hemisphere)
                hemispheres_i.append(int(index_hemisphere))
            
            hemispheres=len(np.unique(hemispheres_i))
            index_hemisphere=np.where(hemispheres_quantity_list==int(hemispheres))[0][0]
            presence[index_hemisphere]=presence[index_hemisphere]+1
        
        return hemispheres_quantity_list, presence/len(filters)
    
     
    
    def calculatePredictivePowerWindowChronology(filters,weights):
        chronology_list=np.array([0,5,10,15,20,25,30,35,40])
        predictive_power=np.zeros([1,len(chronology_list)])[0]
        presence=np.zeros([1,len(chronology_list)])[0]
        
        for i in range(0,len(filters)):
            presence_i=np.zeros([1,len(chronology_list)])[0]
            for j in range(2,len(filters[i])):
                chronology=filters[i][j].split('__')[-2]
                index_chronology=np.where(chronology_list==int(chronology))[0]
                presence_i[index_chronology]=presence_i[index_chronology]+1
                predictive_power[index_chronology]=predictive_power[index_chronology]+abs(weights[i][j-2])
            presence=presence+(presence_i > 0.5).astype(np.int_)
        return chronology_list, presence/len(filters), (predictive_power/np.sum(predictive_power))
    
    
        
    def calculatePredictivePowerFilterTime(filters,weights):
        times_list=Feature.getPreIctalRange()
        predictive_power=np.zeros([1,len(times_list)])[0]
        for i in range(0,len(filters)):
            for j in range(2,len(filters[i])):
                
                time=int(filters[i][j].split('__')[-2])+int(filters[i][0].split('__')[1])
                index_times=np.where(times_list==int(time))[0][0]
                predictive_power[index_times]=predictive_power[index_times]+abs(weights[i][0][j-2])   
        return times_list, (predictive_power/np.sum(predictive_power))  
    
    
    def getFeatureWeightsDistribution(weights):
        features_weights=[]
        for i in range(0,len(weights)):
            for j in range(0,len(weights[i][0])):
                features_weights.append(abs(weights[i][0][j]))                
        return features_weights
    
    
    def getIndividualsNumberPerNumberOfGoodFeatures(weights):
        features_weights=[]
        for i in range(0,len(weights)):
            good_feature_counter=0
            for j in range(0,len(weights[i][0])):
                if abs(weights[i][0][j])>0.01:
                    good_feature_counter=good_feature_counter+1
            features_weights.append(good_feature_counter)            
        return features_weights
    
    
    def getPreIctalLabelsFrequency(filters):
        times_list=Feature.getPreIctalRange()
        frequencies=np.zeros([1,len(times_list)])[0]
       
        for i in range(0,len(filters)):
            time=int(filters[i][0].split('__')[1])
            index_times=np.where(times_list==int(time))[0][0]
            frequencies[index_times]=frequencies[index_times]+1        
        return times_list, frequencies
    
    
    def getFilterSizes(filters):
        filter_sizes=[]
        for i in range(0,len(filters)):
            filter_sizes.append(int(filters[i][1].split('__')[1]))
        return filter_sizes


    
        