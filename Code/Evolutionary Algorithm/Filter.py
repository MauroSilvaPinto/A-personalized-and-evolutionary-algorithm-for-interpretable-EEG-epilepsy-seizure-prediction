"""
Filter class.

Filter class constructs the second-level features (hyper-features)
by windowing the first-level ones. It performs this feature extraction
chronologically


"""


import numpy as np
from scipy.signal import find_peaks
from Feature import Feature
import networkx as nx

class Filter:
    
    # creates a feature with the decoded phenotype, that is
    # calculates the pre-ictal time and the time-delay entries
    # of the filter instead of event moments
    # then calculates the filter size
    # the structures:
    #   pre_ictal time
    #   filter size
    #   filter events
    def createFilter(decoded_phenotype):
        filter=Filter.buildFilterChronology(decoded_phenotype)
        filter=Filter.addFilterLengthToFilter(filter)
        filter=Filter.addPreIctalTimeToFilter(filter,Filter.getPreIctalTime(decoded_phenotype))
        return filter

    # decomposes the filter into pre_ictal time, filter_size and filter_components    
    def decomposeFilter(filter):
        preictal_time=int(filter[0].split("__")[1])
        filter_size=int(filter[1].split("__")[1])
        filter_components=filter[2:]
        
        return preictal_time, filter_size, filter_components
    
    # calculates the preictal time of the decoded phenotype, that is
    # the period more close to a seizure of the phenotype features    
    def getPreIctalTime(decoded_phenotype):  
        return int(min(Filter.getListOfTimeEvents(decoded_phenotype)))
    
    # retrieves the time moments of the events as a list
    def getListOfTimeEvents(decoded_phenotype):
        times=[]
        for feature in decoded_phenotype:
            times.append(feature.split("__")[2])
        return times
    
    # retrieves the first event of the filter, that is, the moment more distant
    # to a seizure
    def getFirstEvent(decoded_phenotype):
        return int(max(Filter.getListOfTimeEvents(decoded_phenotype)))
    
    # builds the filter with the decoded phenotype, that is
    # updates the phenotype with the chronology of the filter
    def buildFilterChronology(decoded_phenotype):
        filter=[]
        for feature in decoded_phenotype:
            filter_component=(Filter.getPreProcessingIndex(feature) + 
                              "__" + Filter.getFeatureOperation(feature) +
                              "__" + str(Filter.getFilterEventMoment(feature,decoded_phenotype)) +
                              "__" + str(Filter.getFeatureEventWindow(feature)))
            
            filter.append(filter_component)
        return filter
    
    
    # adds the preictal time to the filter as the first line
    def addPreIctalTimeToFilter(filter,pre_ictal):
        filter.insert(0,"pre_ictal__"+str(pre_ictal))
        return filter
    
    # adds the filter length to the filter as the first line
    # there was the need to had the first event largest window since it will
    # also contribute to the filter size
    def addFilterLengthToFilter(filter):
        filter_length=0
        for feature_event in filter:
            if ((Filter.getFeatureEventMoment(feature_event) + 
                 Filter.getFeatureEventWindow(feature_event)) > filter_length):
                
                filter_length=(Filter.getFeatureEventMoment(feature_event) + 
                                Filter.getFeatureEventWindow(feature_event))
        
        filter.insert(0,"filter_size__"+str(filter_length))
        return filter
    
    
    
    # adds to the filter length the window event of the first event
    def addToFilterLargestWindow(filter_length, decoded_phenotype):
        largest_window=Filter.findLargestWindowFirstEvent(decoded_phenotype)
        return filter_length+largest_window
    
    # finds the largest window of the first event
    # that is, analysis all window events of the first event
    # and selects the highest
    def findLargestWindowFirstEvent(decoded_phenotype):
        first_event=Filter.getFirstEvent(decoded_phenotype)
        largest_window=0
        for feature_event in decoded_phenotype:
            if Filter.getFeatureEventMoment(feature_event)==first_event:
                if Filter.getFeatureEventWindow(feature_event)>largest_window:
                    largest_window=Filter.getFeatureEventWindow(feature_event)
        return largest_window
            
    # retrieves the event moment in the filter of a certain feature for a given
    # decoded phenotype
    def getFilterEventMoment(feature,decoded_phenotype):
        return (Filter.getFirstEvent(decoded_phenotype)-Filter.getFeatureEventMoment(feature))
    
    #retrieves the feature index  of the preprocessed data  
    def getPreProcessingIndex(feature):
        return feature.split("__")[0]
    
    # retrieves the mathematical operation of the feature
    def getFeatureOperation(feature):
        return feature.split("__")[1]
    
    # retrieves the time moment of a certain feature    
    def getFeatureEventMoment(feature):
        return int(feature.split("__")[2])
    
    # retrieves the window-scale of a certain feature    
    def getFeatureEventWindow(feature):
        return int(feature.split("__")[3])
    
    # retrieves the number of filter components, that is, the number of features
    def getNumberOfFilterComponents(filter_components):
        return len(filter_components)
    
    # gets the step size of the filter in minutes in the iterictal part
    def getStepFilterInMinutesInterictal():
        return 1
    
    # gets the step size of the filter in minutes in the preictal part
    def getStepFilterInMinutesPreictal():
        return 1
    
    # performs the mathematical mean operation
    def performMeanOperation(data,starting_index,size_window):
        return np.mean(data[starting_index:starting_index+size_window])
    
    # performs the mathematical median operation
    def performMedianOperation(data,starting_index,size_window):
        return np.median(data[starting_index:starting_index+size_window])
    
    # performs the mathematical variance operation
    def performVarianceOperation(data,starting_index,size_window):
        return np.var(data[starting_index:starting_index+size_window])
    
    # performs the mathematical integral (area under the curve) operation
    # using trapz rule
    def performIntegralOperation(data,starting_index,size_window):
        return np.trapz(data[starting_index:starting_index+size_window])
    
    # performs a measure of periodism, regarding the mean distance of the locations
    # of peaks
    def performLocPeaksMeanOperation(data,starting_index,size_window):
        peaks,__=find_peaks(data[starting_index:starting_index+size_window])
        return np.mean(np.diff(peaks))
    
    # performs a measure of variance of periodism, regarding the variance of the
    # distance  of the location of peaks
    def performLocPeaksVarianceOperation(data,starting_index,size_window):
        peaks,__=find_peaks(data[starting_index:starting_index+size_window])
        return np.var(np.diff(peaks))
    
    # for a chunk of data and for a filter component, calculate the feature in
    # question represented in the filter component
    def calculateFeature(data, component):
        starting_index=Filter.getFeatureEventMoment(component)
        size_window=Filter.getFeatureEventWindow(component)
        operation=Filter.getFeatureOperation(component)
        
        if operation == "mean":
            value=Filter.performMeanOperation(data,starting_index,size_window)
        elif operation == "median":
            value=Filter.performMedianOperation(data,starting_index,size_window)
        elif operation == "variance": 
            value=Filter.performVarianceOperation(data,starting_index,size_window)
        elif operation == "integral":
            value=Filter.performIntegralOperation(data,starting_index,size_window)
        #elif operation == "loc_pks_mean":
            #value=Filter.performLocPeaksMeanOperation(data,starting_index,size_window)
        #elif operation == "lock_pks_var":
            #value=Filter.performLocPeaksVarianceOperation(data,starting_index,size_window)
            
        return value
    
    # calculate the label regarding the considered pre-ictal time, if it is
    # inter-ictal or pre-ictal
    def calculateLabel(index,seizure_size,filter_size,pre_ictal):
        if (index + filter_size) < (seizure_size - pre_ictal):
            return 0
        else:
            return 1
        
    # performed a moving average filter with the provided size
    def movingAverageFilter(filter_size):
        b=np.ones(filter_size)
        return Filter.normalizeFilter(b) 
    
    # performed a moving average with linear decay filter with the provided size
    def movingLinearDecayFilter(filter_size):
        b=np.linspace(1,filter_size,filter_size)
        return Filter.normalizeFilter(b) 
    
    # performed a moving average with exponential decay filter with the provided size
    def movingExponentialDecayFilter(filter_size):
        b=np.flip(-np.linspace(0,filter_size,filter_size))
        return Filter.normalizeFilter(b) 
    
    # normalize the filter in order for the sum of its contributions is equal to 1
    def normalizeFilter(filter):
        return filter/np.sum(filter)
    
    # after the first classification step, data is post-processed in order to
    # obtain a new one, concerning a provided step size 
    # since there may be more than one seizure, one must make a division between
    # seizures for not making a filter after the end of a seizure and the beginning
    # of another which don't have anything to do with this
    def getPostProcessingData(scores,labels,step):
        
        scores=np.array(scores)
        labels=np.array(labels)
        
        features=[]
        new_labels=[]
        
        indexes_end_seizure=np.where(np.diff(labels)==-1)[0]+1     
        
        #that is, if there is more than one seizure
        if len(indexes_end_seizure)>0:
            # first seizure
            for i in range(0,indexes_end_seizure[0]-step):
                features.append(np.reshape(scores[i:i+step],step))
                new_labels.append(np.clip(np.sum(labels[i:i+step]),0,1))
                
            #for middle seizures
            for i in range(0,len(indexes_end_seizure)-1):
                for j in range(indexes_end_seizure[i],indexes_end_seizure[i+1]-step):
                    features.append(np.reshape(scores[j:j+step],step))
                    new_labels.append(np.clip(np.sum(labels[j:j+step]),0,1))
                            
            #for last seizure
            for i in range(indexes_end_seizure[-1],len(labels)-step):
                features.append(np.reshape(scores[i:i+step],step))
                new_labels.append(np.clip(np.sum(labels[i:i+step]),0,1)) 
            
        # it there is only one seizure  
        else:
            for i in range(0,len(labels)-step):
                features.append(np.reshape(scores[i:i+step],step))
                new_labels.append(np.clip(np.sum(labels[i:i+step]),0,1))
        
        return (np.reshape(np.array(features),[len(features),step]),
                np.array(new_labels))
        
    
    # converts the provided amount of minutes of hours by dividing by 60    
    def convertMinutesInHours(minutes):
        return minutes/60
    
    # calculates the number of triggered alarms
    def calculateNumberOfAlarms(predicted):
        return len(np.where(np.diff(predicted)==1)[0])
    
    # calculates the number of existing seizures
    def calculateNumberOfSeizures(labels):
        return len(np.where(np.diff(labels)==1)[0])
    
    
    def calculateNumberOfFalseAlarmsSurrogate(labels,predicted):
        f_alarms=0
        for i in range(0,len(labels)):
            if labels[i]==0 and predicted[i]==1:
                f_alarms=f_alarms+1
        return f_alarms
    
    # calculates the number of false alarms´
    # first it calculates the number of false alarms in the first seizure
    # then in the remaining ones
    def calculateNumberOfFalseAlarms(labels,predicted):
        indexes_ending_interictal=np.where(np.diff(labels)==1)[0]+1
        indexes_ending_preictal=np.where(np.diff(labels)==(-1))[0]+1

        first_seizure_predicted=predicted[0:indexes_ending_interictal[0]]
        first_seizure=len(np.where(np.diff(first_seizure_predicted)==1)[0])
        
        # if a seizure is immediatelly triggered, the diff function will have no effect
        # we must verify manually the first value
        if (first_seizure_predicted[0]==1):
            first_seizure=first_seizure+1
        
        other_seizures=0
        for i in range(1,len(indexes_ending_interictal)):
            other_seizures_predicted=predicted[indexes_ending_preictal[i-1]:indexes_ending_interictal[i]]
            other_seizures=other_seizures+len(np.where(np.diff(other_seizures_predicted)==1)[0])
            
            # if a seizure is immediatelly triggered, the diff function will have no effect
            # we must verify manually the first value
            if (len(np.where(np.diff(other_seizures_predicted)==1)[0]) ==0 and 
                other_seizures_predicted[0]==1):
                other_seizures=other_seizures+1
                
        return other_seizures+first_seizure
        
    
    
    
    # calculates the number of the well triggered alarms
    # first it calculates the number of triggered alarms in all seizures except
    # the last
    # then it calculates it in the last seizure
    def calculateNumberOfTriggeredSeizures(labels,predicted):
        indexes_ending_interictal=np.where(np.diff(labels)==1)[0]+1
        indexes_ending_preictal=np.where(np.diff(labels)==(-1))[0]+1
        
        other_seizures=0
        for i in range(0,len(indexes_ending_preictal)):
            other_seizures_predicted=predicted[indexes_ending_interictal[i]:indexes_ending_preictal[i]]
            if 1 in other_seizures_predicted:
                other_seizures=other_seizures+1
       
        last_seizure_predicted=predicted[indexes_ending_interictal[-1]-1:]
        if 1 in last_seizure_predicted:
            last_seizure=1
        else:
            last_seizure=0
        
        return other_seizures+last_seizure
    
    def calculateNumberOfTriggeredSeizuresSurrogate(labels,predicted):
        triggered=0
        for i in range(0,len(predicted)):
            if labels[i]==1 and predicted[i]==1:
                triggered=1
        
        return triggered
    
    def calculateDistanceBetweenFilters(filter_a,filter_b):
       filter_a=Filter.assignFilterOrder(filter_a,filter_b)
       events_distances=[]
       for i in range (2,len(filter_a)):
           pre_ictal_a=int(filter_a[0].split("__")[1])
           pre_ictal_b=int(filter_b[0].split("__")[1])
           events_distances.append(Filter.calculateDistanceBetweenComponents(filter_a[i],filter_b[i],
                                                                             pre_ictal_a, pre_ictal_b))
        
       events_distances=np.array(events_distances)
       return (np.sum(events_distances))
        
       
        
        
    def assignFilterOrder(filter_a,filter_b):
        used_indexes=[]
        pre_ictal_a=int(filter_a[0].split("__")[1])
        pre_ictal_b=int(filter_b[0].split("__")[1])
        
        for i in range(2,len(filter_a)):
            component_distances=[]
            for j in range(2,len(filter_b)):
                component_distances.append(Filter.calculateDistanceBetweenComponents(filter_a[i],filter_b[j],
                                                                                     pre_ictal_a, pre_ictal_b))
                
            component_distances=np.array(component_distances)
            indexes_sorted_distances=np.argsort(component_distances)
            
            for j in range(0,len(indexes_sorted_distances)):
                if (indexes_sorted_distances[j]+2) in used_indexes:
                    continue
                else:
                    used_indexes.append(indexes_sorted_distances[j]+2)
                    break
        
        new_filter=[]
        new_filter.append(filter_a[0])
        new_filter.append(filter_a[1])
        for i in range(2,len(filter_a)):
            new_filter.append(filter_a[used_indexes[i-2]])
            
        return new_filter
        
                
            
            

    def calculateDistanceBetweenComponents(component_a,component_b, pre_ictal_a, pre_ictal_b):
        component_differences=[]
        
        component_differences.append(Filter.calculateMathematicalOperatorDistance(component_a,component_b))  
        component_differences.append(Filter.calculateElectrodeDistance(component_a,component_b))
        component_differences.append(Filter.calculateFilterTimeDistance(component_a,component_b,pre_ictal_a, pre_ictal_b))  
        component_differences.append(Filter.calculateCharacteristicDistance(component_a,component_b)) 
        component_differences.append(Filter.calculateWindowLengthDistance(component_a,component_b))
        
        component_differences=np.array(component_differences)
        return (np.sum(component_differences))
    
    
    
    def calculateMathematicalOperatorDistance(component_a,component_b):
        operator_a=component_a.split('__')[1]
        operator_b=component_b.split('__')[1]
        
        if operator_a==operator_b:
            return 0
        else:
            return 1
        
        
    def calculateElectrodeDistance(component_a,component_b):
        electrode_a=component_a.split('_')[0]
        electrode_b=component_b.split('_')[0]
        
        brain_graph=Feature.getElectrodesGraph()
        paths=list(nx.all_shortest_paths(brain_graph,
                                         source=electrode_a,
                                         target=electrode_b))      
        path=paths[np.random.choice(np.arange(0,len(paths)))]
          
        return (len(path)-1)
    
    def calculateWindowLengthDistance(component_a,component_b):
        window_a=component_a.split('__')[-1]
        window_b=component_b.split('__')[-1]

        window_length_range=Feature.getWindowLengthRange()
        window_a_index=np.where(window_length_range==int(window_a))[0][0]
        window_b_index=np.where(window_length_range==int(window_b))[0][0]
        
        return abs(window_a_index-window_b_index)
    
    
    def calculateFilterTimeDistance(component_a, component_b,pre_ictal_a, pre_ictal_b):
        time_a=int(component_a.split('__')[-2])+pre_ictal_a
        time_b=int(component_b.split('__')[-2])+pre_ictal_b
        
        preictal_time_range=Feature.getPreIctalRange()
        time_a_index=np.where(preictal_time_range==int(time_a))[0][0]
        time_b_index=np.where(preictal_time_range==int(time_b))[0][0]
        return abs(time_a_index-time_b_index)
    
    
    def isCharacteristicWave(characteristic):
        return (characteristic in Feature.getWavesList())
    
    
    def calculateCharacteristicDistance(component_a, component_b):
        characteristic_a=component_a.split('__')[0].split('_',1)[-1]
        characteristic_b=component_b.split('__')[0].split('_',1)[-1]
        
        if (Filter.isCharacteristicWave(characteristic_a) and 
            Filter.isCharacteristicWave(characteristic_b)):
            
            index_a=Feature.getWavesList().index(characteristic_a)
            index_b=Feature.getWavesList().index(characteristic_b)
            
            return abs(index_a-index_b)
        
        elif (not Filter.isCharacteristicWave(characteristic_a) and 
            not Filter.isCharacteristicWave(characteristic_b)):
            if characteristic_a==characteristic_b:
                return 0
            else:
                return 1
        
        else:
            return 1
                
        
        
    
                
        
        
        