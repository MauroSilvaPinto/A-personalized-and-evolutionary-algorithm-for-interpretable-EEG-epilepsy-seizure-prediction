"""
Feature Class. Maybe it should be called Hyper-feature

Each hyper-feature contains several genes:
    one gene for mathematical operator
    one gene for electrode
    one gene for pre-ictal period (named in the paper as delay)
    one gene for the first-level feature (characteristic)
    one gene for the band-wave first-level feature
    one gene for a non band-wave first-level feature
    one gene for a window length

    ps: electrodes are in the old nomenclature of 10-20 eeg system
    
    T3 is now T7.
    T4 is now T8.
    T5 is now P7.
    T6 is now P8.
    
"""


import os
import random
import numpy as np
import networkx as nx
import copy

class Feature:

    # initialize a random Feature instance, 
    # where all attributes are initialized randomly
    def __init__(self,path):  
        self.path=path
        
        self.mathematical_operator=Feature.generateRandomMathematicalOperator();
        self.electrode=self.generateRandomElectrode()
        self.preictal_time=Feature.generateRandomPreIctalTime()
        self.characteristic=Feature.generateRandomCharacteristic()
        self.wave=Feature.generateRandomWave()
        self.window_length=Feature.generateRandomWindowLength()
        self.wave_non_wave=Feature.generateRandomWaveNonWave()
        

    # gets the preprocessing labels to find the index with the electrode and
    # correspondent characteristic, to retrieve the data on the way it is stored
    def getPreprocessingLabels(self):
        os.chdir(self.path)
        with open("preprocessing_labels.txt", "r") as f:
            line = f.readlines()
        return line
        #return(open("preprocessing_labels.txt", "r").readlines())
        
        
    # gets the list of extracted electrodes of the used 
    # pre-process signals. this is known by the provided path location where the
    # labels are present in the same directory
    def getElectrodesList(self):
        electrodes=[]    
        
        for characteristic in self.getPreprocessingLabels():
            if (Feature.getElectrodeName(characteristic) not in electrodes and 
                Feature.getElectrodeName(characteristic) != "SP1" and 
                Feature.getElectrodeName(characteristic)!="SP2" and
                Feature.getElectrodeName(characteristic) !="RS" and
                Feature.getElectrodeName(characteristic) !="T1" and
                Feature.getElectrodeName(characteristic) !="T2"):
                electrodes.append(Feature.getElectrodeName(characteristic))
        return electrodes
    
    # retrieves the electrode of a preprocessing label
    def getElectrodeName(characteristic):
        return characteristic.split("_")[0]
    
    # provides the mathematical operators' list (this was not performed autoamtically)
    # due to the fact of wave/non wave origin
    def getMathematicalOperatorsList():
        return ["median", "mean", "variance", "integral"] # "loc_pks_mean", "lock_pks_var"]  
    
    # provides the characteristic's list
    def getCharacteristicsList():
        return ["mean_freq","band_power","medium_intensity",
                "medium_intensity_unormalized","variance"]
    
    # provides the waves list
    def getWavesList():
        return ["delta","theta","alpha","beta","gamma_1","gamma_2","gamma_3"]
    
    # defines the pre-ictal range
    def getPreIctalRange():
        return np.arange(30,50,5)
    
    # defines the window length range
    def getWindowLengthRange():
        return np.array([1, 5, 10, 15, 20])
    
    # selects a random Wave or Non Wave origin
    def generateRandomWaveNonWave():
        return random.randint(0,1)
    
    # selects a random mathematical operator
    def generateRandomMathematicalOperator():
        return random.randint(0,len(Feature.getMathematicalOperatorsList())-1)
    
    # selects a random electrode
    def generateRandomElectrode(self):
        return random.randint(0,len(self.getElectrodesList())-1)
    
    # selects a random pre-ictal time
    def generateRandomPreIctalTime():
        return random.choice(Feature.getPreIctalRange())
    
    def generateRandomWaveOrigin():
        return random.randint(0,1)
    
    # selects a random characteristic
    def generateRandomCharacteristic():
        return random.randint(0,len(Feature.getCharacteristicsList())-1)
    
    # selects a random wave   
    def generateRandomWave():
        return random.randint(0,len(Feature.getWavesList())-1)
    
    # selects a random window length
    def generateRandomWindowLength():
        return random.choice(Feature.getWindowLengthRange())
    
    # prints the attributes of the feature
    def printFeature(self):
        print("Electrode: "+ str(self.electrode))
        print("Wave: "+ str(self.wave))
        print("Characteristic: "+ str(self.characteristic))
        print("Mathematical Operator: "+ str(self.mathematical_operator))
        print("Pre-Ictal Time: "+ str(self.preictal_time))
        print("Window Length: "+ str(self.window_length))
        
        
    
    # returns a boolean, True if the feature is concerning a wave and false if
    # otherwise
    def isWave(self):
        return self.wave_non_wave==1
    
    # returns as string the name of the electrode 
    def decodeElectrode(self):
        return self.getElectrodesList()[self.electrode]      
    
    # returns the wave of the feature if it is a wave, if otherwise, of the
    # name of the characteristic
    def decodeWaveAndCharacteristic(self):
        if self.isWave():
            return self.decodeWave()
        else:
            return self.decodeCharacteristic()
    
    # returns as string the name of the characteristic of the feature
    def decodeCharacteristic(self):
        return Feature.getCharacteristicsList()[self.characteristic]
    
    # returns as string the name of the wave
    def decodeWave(self):
        return Feature.getWavesList()[self.wave]
    
    # returns as string the name of the mathematical operator
    def decodeMathematicalOperator(self):
        return Feature.getMathematicalOperatorsList()[self.mathematical_operator]
    
    # prints as a string the meaning of the feature    
    def printDecodedFeature(self):
        return (self.decodeElectrode()+ "_" + self.decodeWaveAndCharacteristic()+
                "_" + self.decodeMathematicalOperator()+"__"+str(self.preictal_time)+ 
                   "_" + str(self.window_length))
    
    # returns the index of the feature concerning the preprocessing labels,
    # in order to retrieve the seizure data with that index
    # [0] is to return an integer and not an arary integer,
    # note: moreover, in "intensity", "intensity_unormalized" was also picked
    # so this resolves that problem
    def getIndexPreprocessingLabels(self):
        return ([i for i, labels in enumerate(self.getPreprocessingLabels()) if
                self.decodeElectrode() in labels and 
                self.decodeWaveAndCharacteristic() in labels][0])
    
    # returns as string the phenotype of the feature, that is, the index
    # location of the seizure data, the mathematical operator to use
    # and the preictal time along with the window-length to use        
    def getDecodedPhenotype(self):
        return (str(self.getIndexPreprocessingLabels())+ "__" +
                self.decodeMathematicalOperator()+ "__" +
                str(self.preictal_time)+ "__" +
                str(self.window_length))
    
    # mutates a feature randomly
    # randomly, it selects on which part a mutation will be performed:
    #   on the mathematical operator
    #   on the characteristic
    #   on the electrode
    #   on the preictal time
    #   on the window length    
    def mutate(self):
        elements=[1,2,3,4,5]
        weights=Feature.getMutationProbabilityWeights()
        
        mutation_part=np.random.choice(elements, p=weights)
               
        if mutation_part==1:
            self.mutateMathematicalOperator()
        elif mutation_part==2:
            self.mutateCharacteristic()
        elif mutation_part==3:
            self.mutateElectrode()
        elif mutation_part==4:
            self.mutatePreIctalTime()
        elif mutation_part==5:
            self.mutateWindowLength()
            
    def getMutationProbabilityWeights():
        probs=np.zeros(5)
        probs[0]=len(Feature.getMathematicalOperatorsList())
        probs[1]=len(Feature.getCharacteristicsList())
        probs[2]=Feature.getElectrodesGraph().number_of_nodes()
        probs[3]=len(Feature.getPreIctalRange())
        probs[4]=len(Feature.getWindowLengthRange())
        
        return probs/np.sum(probs)
        
        
   
    # mutates the mathematical operator by selecting randomly another operator
    # while the previous one mathematical operator is still being chosen 
    # randomly, the task is repeated untill a mutation occurs
    def mutateMathematicalOperator(self):
        new_mathematical_operator=Feature.generateRandomMathematicalOperator()
        while new_mathematical_operator==self.mathematical_operator:
            new_mathematical_operator=Feature.generateRandomMathematicalOperator()
        self.mathematical_operator=new_mathematical_operator
    
    # mutates the characteristic
    # first, a flip coin operation is performed to decide if the mutation is
    # related with the wave or non-wave origin
    # if it is a mutation wave-type, a mutation will occur regarding the wave
    # if it is a mutation non-wave-type, a mutation will occur regarding the non-wave
    def mutateCharacteristic(self):
        wave_non_wave_mutated=Feature.generateRandomWaveOrigin()
       
        if wave_non_wave_mutated == 1 and self.isWave():
            self.mutateWave()
            
        if wave_non_wave_mutated ==1 and not self.isWave():
            self.wave_non_wave=wave_non_wave_mutated
            self.mutateWave()
            
        if wave_non_wave_mutated == 0 and self.isWave():
            self.wave_non_wave=wave_non_wave_mutated
            self.characteristic=Feature.generateRandomCharacteristic()
        
        if wave_non_wave_mutated == 0 and not self.isWave():
            self.characteristic=Feature.generateRandomCharacteristic()
    
    # mutates the wave, a flip a coin operation is also performed to choose
    # if a wave will be mutated to a higher or lower frequency band    
    def mutateWave(self):
        up_down_mutation=random.randint(0,1)
        if up_down_mutation==0:
            self.mutateWaveDown()
        if up_down_mutation==1:
            self.mutateWaveUp()
    
    # mutates a frequency band to a lower frequency range, unless the bottom
    # frequency band is reached. in that case, the mutation occurs upwards        
    def mutateWaveDown(self):
        if self.wave==0:
            self.mutateWaveUp()
        else:
            self.wave=self.wave-1
            
    # mutates a frequency band to a higher frequency range, unless the bottom
    # frequency band is reached. in that case, the mutation occurs downaards           
    def mutateWaveUp(self):
        if self.wave==len(Feature.getWavesList())-1:
            self.mutateWaveDown()
        else:
            self.wave=self.wave+1
    
    # mutates the pre-ictal time, where a flip a coin operation is performed to
    # decide if the mutation will occur downards or uppwards        
    def mutatePreIctalTime(self):
        up_down_mutation=random.randint(0,1)
        if up_down_mutation==0:
            self.mutatePreIctalDown()
        if up_down_mutation==1:
            self.mutatePreIctalUp()
    
    # the preictal time will be mutated downwards. if the lower limit of pre-ictal
    # time is reached, the mutation will occur upwards       
    def mutatePreIctalDown(self):
        if self.preictal_time==Feature.getPreIctalRange()[0]:
            self.mutatePreIctalUp()
        else:
            old_index=np.where(Feature.getPreIctalRange()==self.preictal_time)[0][0]
            self.preictal_time=Feature.getPreIctalRange()[old_index-1]
        
    # the preictal time will be mutated upwards. if the lower limit of pre-ictal
    # time is reached, the mutation will occur downwards       
    def mutatePreIctalUp(self):
        if self.preictal_time==Feature.getPreIctalRange()[len(Feature.getPreIctalRange())-1]:
            self.mutatePreIctalDown()
        else:
            old_index=np.where(Feature.getPreIctalRange()==self.preictal_time)[0][0]
            self.preictal_time=Feature.getPreIctalRange()[old_index+1]    
    
    # mutates the window length time, where a flip a coin operation is performed to
    # decide if the mutation will occur downards or uppwards 
    def mutateWindowLength(self):
        up_down_mutation=random.randint(0,1)
        if up_down_mutation==0:
            self.mutateWindowLengthDown()
        if up_down_mutation==1:
            self.mutateWindowLengthUp()
            
    # the window length time will be mutated downwards. if the lower limit of window length
    # time is reached, the mutation will occur upwards     
    def mutateWindowLengthDown(self):
        if self.window_length==Feature.getWindowLengthRange()[0]:
            self.mutateWindowLengthUp()
        else:
            old_index=np.where(Feature.getWindowLengthRange()==self.window_length)[0][0]
            self.window_length=Feature.getWindowLengthRange()[old_index-1]    
            
   
    # the window length time will be mutated upwards. if the lower limit of window length
    # time is reached, the mutation will occur downwards
    def mutateWindowLengthUp(self):
        if self.window_length==Feature.getWindowLengthRange()[len(Feature.getWindowLengthRange())-1]:
            self.mutateWindowLengthDown()
        else:
            old_index=np.where(Feature.getWindowLengthRange()==self.window_length)[0][0]
            self.window_length=Feature.getWindowLengthRange()[old_index+1]   
            
    # mutates an electrode by listing the adjacent electrodes of the current one
    # and by selecting randomnly a neighbor one    
    def mutateElectrode(self):
        brain_graph=Feature.getElectrodesGraph()
        current_electrode=self.getElectrodesList()[self.electrode]
        electrode_neighbors=list(brain_graph.neighbors(current_electrode))
        mutated_electrode=np.random.choice(electrode_neighbors)
        
        self.electrode=self.getElectrodesList().index(mutated_electrode)
     
    
    # creates a new feature through the recombination of 2 features
    #   all parts are recombined: the mathematical operator, the electrode,
    #                             the characteristic, the window length and the
    #                             the preictal time
    def recombinateFeature(parent_1_feature, parent_2_feature):
        new_feature=copy.deepcopy(parent_1_feature)
        new_feature.recombineMathematicalOperator(parent_1_feature, parent_2_feature)
        new_feature.recombineElectrode(parent_1_feature, parent_2_feature)
        new_feature.recombineCharacteristic(parent_1_feature, parent_2_feature)
        new_feature.recombineWindowLength(parent_1_feature, parent_2_feature)
        new_feature.recombinePreictalTime(parent_1_feature, parent_2_feature)
        
        return new_feature
     
    

    # recombination of the mathematical operator of two features
    # picks one of the mathematical operators of the two parents
    def recombineMathematicalOperator(self,parent_1_feature,parent_2_feature):
        self.mathematical_operator=np.random.choice([parent_1_feature.mathematical_operator,
                                                     parent_2_feature.mathematical_operator])
    
    
    
    # recombination of the electrode of two features
    #   - when the electrode is the same on both parents, the electrode remains 
    #     the same
    #   - when the electrodes are adjacent, one is chosen randomnly
    #   - in the remaining cases, the shortest paths from one electrode to the 
    #     other. one path is randomnly chosen and then one electrode from that
    #     path is randomnly chosen
    def recombineElectrode(self,parent_1_feature,parent_2_feature):
        brain_graph=Feature.getElectrodesGraph()
        
        electrode_father_1=self.getElectrodesList()[parent_1_feature.electrode]
        electrode_father_2=self.getElectrodesList()[parent_2_feature.electrode]
        
        if electrode_father_1==electrode_father_2:
            self.electrode=parent_1_feature.electrode
            
        
        else:
            paths=list(nx.all_shortest_paths(brain_graph,
                                         source=electrode_father_1,
                                         target=electrode_father_2))
            path=paths[np.random.choice(np.arange(0,len(paths)))]
            
            if len(path)==2:
                self.electrode=np.random.choice([parent_1_feature.electrode,
                                                     parent_2_feature.electrode])    
            else:
                recombined_electrode=np.random.choice(np.array(path[1:-1]))
                self.electrode=self.getElectrodesList().index(recombined_electrode)
                
    

    # recombination of the non-wave characteristic operator of two features
    # picks one of the non-wave characteristic of the two parents       
    def recombineNonWave(self,parent_1_feature,parent_2_feature):
        self.characteristic=np.random.choice([parent_1_feature.characteristic,
                                              parent_2_feature.characteristic])
    
    
    # recombination of the electrode of two features
    #   - when the electrode is the same on both parents, the electrode remains 
    #     the same
    #   - when the electrodes are adjacent, one is chosen randomnly
    #   - in the remaining cases, the shortest paths from one electrode to the 
    #     other. one path is randomnly chosen and then one electrode from that
    #     path is randomnly chosen    
    def recombineWave(self,parent_1_feature,parent_2_feature):
        if parent_1_feature.wave==parent_2_feature.wave:
            self.wave=parent_1_feature.wave
        else:
            if abs(parent_1_feature.wave-parent_2_feature.wave)<2:
                self.wave=np.random.choice([parent_2_feature.wave,
                                                     parent_2_feature.wave])
            else:
                list_waves=[parent_1_feature.wave,parent_2_feature.wave]
                recombined_wave=np.random.choice(np.arange(min(list_waves)+1,max(list_waves)))
                self.wave=recombined_wave
                 
    
    # recombines the characteristic of two features
    #   - if the two features are wave origin type
    #       a wave recombination is performed
    #   - if the two features are non wave origin type
    #       a non-wave recombination is performed
    #   - else
    #       a coin-flip operation is performed to decide whether to perform
    #       either a wave recombination or a non-wave recombination           
    def recombineCharacteristic(self,parent_1_feature, parent_2_feature):
        if parent_1_feature.isWave() and parent_2_feature.isWave():
            self.recombineWave(parent_1_feature, parent_2_feature)
            
        if not parent_1_feature.isWave() and not parent_2_feature.isWave():
            self.recombineNonWave(parent_1_feature,parent_2_feature)
            
        else:
            wave_non_wave=Feature.generateRandomWaveOrigin()
            if wave_non_wave==1:
                self.wave_non_wave=wave_non_wave
                self.recombineWave(parent_1_feature,parent_2_feature)
            elif wave_non_wave==0:
                self.wave_non_wave=wave_non_wave
                self.recombineNonWave(parent_1_feature,parent_2_feature)
        
    # recombines the preictal time of two features
    #   - if the preictal times of the features are equal
    #       the recombinated preictal time is the same
    #   - if the preictal times of the features are consecutive
    #       one of the parents preictal times is chosen randomnly
    #   - else
    #       a preictal time between both parent preictal times is randomnly chosen
    def recombinePreictalTime(self,parent_1_feature,parent_2_feature):
        index_parent_1_preictal=np.where(Feature.getPreIctalRange()==parent_1_feature.preictal_time)[0][0]
        index_parent_2_preictal=np.where(Feature.getPreIctalRange()==parent_2_feature.preictal_time)[0][0]
        
        list_indexes=[index_parent_1_preictal,index_parent_2_preictal]
        
        if index_parent_1_preictal==index_parent_2_preictal:
            self.preictal_time=parent_1_feature.preictal_time    
        else:
            if abs(index_parent_1_preictal-index_parent_2_preictal)<2:
                self.preictal_time=np.random.choice([parent_1_feature.preictal_time,
                                                     parent_2_feature.preictal_time])
            else:
                recombined_index=np.random.choice(np.arange(min(list_indexes)+1,max(list_indexes)))
                self.preictal_time=Feature.getPreIctalRange()[recombined_index]
                               
    # recombines the window length of the two features
    #   - if the window lengths of the features are equal
    #       the recombinated window length is the same
    #   - if the window lengths of the features are consecutive
    #       one of the parents window length is chosen randomnly
    #   - else
    #       a window length between both parent window lengths is randomnly chosen
    def recombineWindowLength(self,parent_1_feature, parent_2_feature):
        index_parent_1_window=np.where(Feature.getWindowLengthRange()==parent_1_feature.window_length)[0][0]
        index_parent_2_window=np.where(Feature.getWindowLengthRange()==parent_2_feature.window_length)[0][0]
        
        list_indexes=[index_parent_1_window,index_parent_2_window]
        
        if index_parent_1_window==index_parent_2_window:
            self.preictal_time=parent_1_feature.preictal_time    
        else:
            if abs(index_parent_1_window-index_parent_2_window)<2:
                self.window_length=np.random.choice([parent_1_feature.window_length,
                                                     parent_2_feature.window_length])
            else:
                recombined_index=np.random.choice(np.arange(min(list_indexes)+1,max(list_indexes)))
                self.window_length=Feature.getWindowLengthRange()[recombined_index]
    
    
    # builds a graph of the brain where the nodes are the electrodes and the
    # edges are connections betwen adjacent electrodes in terms of localization                   
    def getElectrodesGraph():
        
        G=nx.Graph()

        electrodes_list=['C3','C4','CZ','F3','F4','F7', 'F8','FP1','FP2',
                         'FZ','O1','O2','P3','P4','PZ', 'T3','T4','T5','T6']
        
        for electrode in electrodes_list:
            G.add_node(electrode)
                       
        G.add_edge("FP1","FP2")
        G.add_edge("FP1","FZ")
        G.add_edge("FP1","F3")
        G.add_edge("FP1","F7")
        G.add_edge("FP2","FZ")
        G.add_edge("FP2","F4")
        G.add_edge("FP2","F8")
        G.add_edge("F7","F3")
        G.add_edge("F7","C3")
        G.add_edge("F7","T3")      
        G.add_edge("F3","T3")
        G.add_edge("F3","C3")
        G.add_edge("F3","CZ")
        G.add_edge("F3","FZ")        
        G.add_edge("FZ","CZ")
        G.add_edge("FZ","F4")
        G.add_edge("FZ","C4")     
        G.add_edge("F4","CZ")
        G.add_edge("F4","F8")
        G.add_edge("F4","T4")
        G.add_edge("F4","C4")     
        G.add_edge("F8","T4")
        G.add_edge("F8","C4")     
        G.add_edge("T3","T5")
        G.add_edge("T3","C3")
        G.add_edge("T3","P3")      
        G.add_edge("C3","T5")
        G.add_edge("C3","P3")
        G.add_edge("C3","CZ")
        G.add_edge("C3","PZ")
        G.add_edge("C3","FZ")     
        G.add_edge("CZ","P3")
        G.add_edge("CZ","PZ")
        G.add_edge("CZ","C4")
        G.add_edge("CZ","P4")       
        G.add_edge("C4","PZ")
        G.add_edge("C4","P4")
        G.add_edge("C4","T6")
        G.add_edge("C4","T4")       
        G.add_edge("T4","T6")
        G.add_edge("T4","P4")      
        G.add_edge("T5","P3")
        G.add_edge("P3","PZ")
        G.add_edge("P4","PZ")
        G.add_edge("P4","T6")
        G.add_edge("O1","T5")
        G.add_edge("O1","P3")
        G.add_edge("O1","PZ")
        G.add_edge("O1","O2")        
        G.add_edge("O2","PZ")
        G.add_edge("O2","P4")
        G.add_edge("O2","T6")
        
        return G
    
    
    def distanceBetweenMathematicalOperator(feature_1, feature_2):
        if feature_1.mathematical_operator==feature_2.mathematical_operator:
            return 0
        else:
            return 1
        

    def distanceBetweenElectrode(feature_1,feature_2):
        electrode_1=feature_1.getElectrodesList()[feature_1.electrode]
        electrode_2=feature_1.getElectrodesList()[feature_2.electrode]
        
   
        brain_graph=Feature.getElectrodesGraph()
        paths=list(nx.all_shortest_paths(brain_graph,
                                         source=electrode_1,
                                         target=electrode_2))      
        path=paths[np.random.choice(np.arange(0,len(paths)))]
          
        return (len(path)-1)
    
    def distanceBetweenWindowLength(feature_1,feature_2):
        window_1=feature_1.window_length
        window_2=feature_2.window_length

        window_length_range=Feature.getWindowLengthRange()
        window_1_index=np.where(window_length_range==int(window_1))[0][0]
        window_2_index=np.where(window_length_range==int(window_2))[0][0]
        
        return abs(window_1_index-window_2_index)
    
    def distanceBetweenPreIctalTime(feature_1,feature_2):
        time_1=feature_1.preictal_time
        time_2=feature_2.preictal_time
        
        preictal_time_range=Feature.getPreIctalRange()
        time_1_index=np.where(preictal_time_range==int(time_1))[0][0]
        time_2_index=np.where(preictal_time_range==int(time_2))[0][0]
        return abs(time_1_index-time_2_index)
    
    
    def distanceBetweenCharacteristic(feature_1,feature_2):
        if (feature_1.wave_non_wave==1 and feature_2.wave_non_wave==1):
            index_1=feature_1.wave
            index_2=feature_2.wave
            return abs(index_1-index_2)       
        elif (not feature_1.wave_non_wave==1 and not feature_2.wave_non_wave==1):
            if feature_1.characteristic==feature_2.characteristic:
                return 0
            else:
                return 1 
        else:
            return 1
    
    
    
    
    def calculateDistanceBetweenFeatures(feature_1,feature_2):
        distances=[]
        distances.append(Feature.distanceBetweenMathematicalOperator(feature_1,feature_2))
        distances.append(Feature.distanceBetweenElectrode(feature_1,feature_2))
        distances.append(Feature.distanceBetweenCharacteristic(feature_1,feature_2))
        distances.append(Feature.distanceBetweenWindowLength(feature_1,feature_2))
        distances.append(Feature.distanceBetweenPreIctalTime(feature_1,feature_2))
        
        distances=np.array(distances)
        
        return np.sqrt(np.sum(distances**2))
        
        
        
                    