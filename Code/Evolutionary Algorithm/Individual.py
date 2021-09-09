"""
Class Individual: it represents an individual of
the population.

Each individual has a fitness value, and a set of hyper-features

"""


from Feature import Feature
import numpy as np
import random
import copy 

class Individual:
    
    # initiates an individual, consisting on a path of the data to retrieve
    # the electrode and other feature informations
    # it also contains the number of features, that is fixed
    # and it contains the list of features that constitute the individual
    # it also contains the objectives fitness value
    def __init__(self,number_features,path):
         self.path=path
         self.number_features=number_features
         
         self.features=[]
         self.generateRandomFeatures()
         self.fitness=0
         
    # generates random features for this instance in equal number to the
    # number of features provided in the initialization moment     
    def generateRandomFeatures(self):
         for i in range(0,self.number_features):
            self.addFeatureToIndividual(self.generateNewFeature())
    
    # prints the genotype of the individual, that is
    # the genotype of all features         
    def printIndividualGenotype(self):
        for i in range (0,len(self.features)):
            print("Feature "+str(i)+ "\n")
            self.features[i].printFeature()
            print("\n")
    
    # updates the fitness attribute of the individual
    def updateFitness(self,fitness):
        self.fitness=fitness
    
    # generates randomnly a new feature        
    def generateNewFeature(self):
        return Feature(self.path)
    
    # appends the provided feature to the list of individuals
    def addFeatureToIndividual(self,feature):
        self.features.append(feature)

    # prints the feature as a string for a user to understand in what it 
    # consists
    def printDecodedFeatures(self):
       for i in range (0,self.number_features):
            print(self.features[i].printDecodedFeature())
            print(self.features[i].getIndexPreprocessingLabels())
    
    # prints the phenotype of all features       
    def printDecodedPhenotype(self):
        for feature in self.features:
            print(feature.getDecodedPhenotype())
    
    # returns the phenotype of all features as a list where each element is the
    # phenotype of a feature
    def getDecodedPhenotype(self):
        phenotype=[]
        for feature in self.features:
            phenotype.append(feature.getDecodedPhenotype())
        return phenotype
    
    # performs a copy of the individual
    # performs a mutation in the new individual
    # returns the mutated individual
    def mutate(self):
        new_individual=copy.deepcopy(self)
        new_individual.features[random.randint(0,self.number_features-1)].mutate()
        new_individual.fitness=0
        
        return new_individual

    def rearrangeFeaturesAccordingTo(self, other_parent):
        copy_individual=copy.deepcopy(self)
        used_indexes=[]
        for i in range(0,len(self.features)):
            feature_distances=[]
            for j in range(0,len(self.features)):
                feature_distances.append(Feature.calculateDistanceBetweenFeatures(self.features[i],
                                                                               other_parent.features[j]))
                
            feature_distances=np.array(feature_distances)
            indexes_sorted_distances=np.argsort(feature_distances)
            
            for j in range(0,len(indexes_sorted_distances)):
                if (indexes_sorted_distances[j]) in used_indexes:
                    continue
                else:
                    used_indexes.append(indexes_sorted_distances[j])
                    break 
        for i in range(0,len(self.features)):
            self.features[i]=copy_individual.features[used_indexes[i]]
                
    

    def recombination(parent_1,parent_2):
        new_individual=copy.deepcopy(parent_1)
        parent_1.rearrangeFeaturesAccordingTo(parent_2)
        for i in range(len(parent_1.features)):
            new_individual.features[i]=Feature.recombinateFeature(parent_1.features[i],
                                   parent_2.features[i])
        new_individual.fitness=0
        return new_individual