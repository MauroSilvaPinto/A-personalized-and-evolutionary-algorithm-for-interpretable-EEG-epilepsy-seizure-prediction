"""
Poplulation class.
It represents the population of the EA.

A population is constituted by a set of individuals,
which is changing over generations.

"""


from Individual import Individual
from Filter import Filter
import matplotlib.pyplot as plt
import numpy as np
import random
import os


class Population:
    
    # creates population object
    # each population has a set of individuals
    # in a list, it is also stored the current fitness of the population individuals
    # a list with the recorded evolution (fitness generations) for each generation:
    #   the min fitness, mean fitness and max fitness
    # the database is also an attribute, since is necessary to evaluate the fitness
    # function
    def __init__(self,population_size,number_features,path,database):
        self.individuals=[]
        self.generateNindividuals(population_size,number_features,path)
        self.individuals_fitness=[]        
        self.fitness_generations=[]
        self.database=database
        self.population_size=population_size
        self.path=path
        self.best_solution_fitness=[]
        self.best_solution_individual=[]
        self.number_features=number_features
        
        
    def generateRandomOffspring(self):
        offspring=[]
        for i in range(0,self.population_size):
            individual=Population.generateSingleIndividual(self.number_features,
                                                           self.path)
            offspring.append(individual)
        return offspring
        
    
    # generates individuals for the population as a random initialization
    # according to a given population_size, where each individual has a certain
    # number of feature (that is, filter components)
    def generateNindividuals(self,population_size,number_features,path):
        for i in range(0,population_size):
            individual=Population.generateSingleIndividual(number_features,path)
            self.addIndividualToPopulation(individual)
    
    # updates the list of the fitness of the individuals fitness in the population
    # during all evolution        
    def updateIndividualsFitness(self):
        self.individuals_fitness.append(self.getIndividualsFitness())
    
    # retrieves a numpy array with all the current individuals fitness present in the
    # population during the evolution
    def getIndividualsFitness(self):
        fitness_individuals=[]
        for individual in self.individuals:
            fitness_individuals.append(individual.fitness)
        return np.row_stack(fitness_individuals)
    
    # generates a new individual randomly        
    def generateSingleIndividual(number_features,path):
        return Individual(number_features,path)
    
    # adds a given individual to the population
    def addIndividualToPopulation(self, individual):
        self.individuals.append(individual)         
    
    # evaluates the population fitness, that is, iterates all individuals and
    # for each, the fitness value is calculated and provided
    # afterwards, the list of the population fitness and the list of fitness
    # generation metrics are updated
    #
    # note: this function is for initializing the population, in terms of the
    # first evaluation when the individuals are created for the first time
    def evaluatePopulationFitness(self):

        for individual in self.individuals:
            filter=Filter.createFilter(individual.getDecodedPhenotype())
            fitness = self.database.applyFilterToDatabaseWithPast(filter,"validation_data",
                                                          "fitness","mean",0)

            individual.updateFitness(fitness)
        
        self.updateIndividualsFitness()
        self.updateGenerationFitnessMetrics()            
    
    # prints the list of the individuals fitness present in the population        
    def printFitness(self):
        print(self.individuals_fitness)      
        
    # returns the fitness metrics for the current generation:
    #   min fitness, mean fitness and max fitness
    def getGenerationFitnessMetrics(self):
        return [np.min(self.getIndividualsFitness()),
                np.mean(self.getIndividualsFitness()),
                np.max(self.getIndividualsFitness())]
    
    # updates in the population object the fitness metrics of the current generation
    def updateGenerationFitnessMetrics(self):
        self.fitness_generations.append(self.getGenerationFitnessMetrics())
        
    def updateBestIndividual(self):
        best_fitness, best_individual=self.getBestIndividual()
        if (best_fitness>self.best_solution_fitness or 
            self.best_solution_fitness == []):
            
            self.best_solution_fitness=best_fitness
            self.best_solution_individual=best_individual
        
        
        
    def getBestIndividual(self):
        best_individual=self.individuals[0]
        best_fitness=self.individuals[0].fitness
        
        for individual in self.individuals:
            if individual.fitness>best_fitness:
                best_individual=individual
                best_fitness=individual.fitness
        return best_fitness, best_individual
        
        
    # plots the evolution of the population, that is, for each generation, 
    # plots:
    #       the min fitness as red circles
    #       the mean fitness as a blue line
    #       the max fitness as green triangles
    def plotEvolutionMaxMeanMin(self):
        generations=np.linspace(1,len(self.fitness_generations),
                                len(self.fitness_generations))
        fitness_generations=np.array(self.fitness_generations)
        fig, ax = plt.subplots()
        ax.plot(generations, fitness_generations[:,0],'ro',
                generations, fitness_generations[:,1],'b--',
                generations, fitness_generations[:,2],'g^')
        
        ax.set(xlabel='generation', ylabel='Fitness - AUC',
        title='Min, Mean and Max Fitness values throughout Generations')
        ax.grid()
        plt.legend(('Min', 'Mean', 'Max'),
           loc='lower right')
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    #plot the min max mean plot but with the data after it was saved
    def plotEvolutionMaxMeanMinSavedData(data_generations):
        fitness_min=np.min(data_generations,axis=0)
        fitness_mean=np.mean(data_generations,axis=0)
        fitness_max=np.max(data_generations,axis=0)
        
        generations=np.linspace(1,np.shape(data_generations)[1],
                                np.shape(data_generations)[1])
        fig, ax = plt.subplots()
        ax.plot(generations, fitness_min,'ro',
                generations, fitness_mean,'b--',
                generations, fitness_max,'g^')
        
        ax.set(xlabel='generation', ylabel='Fitness - AUC',
        title='Min, Mean and Max Fitness values throughout Generations')
        ax.grid()
        plt.legend(('Min', 'Mean', 'Max'),
           loc='lower right')
     
        fig_to_be_handled=plt.gcf()
        return fig_to_be_handled
    
    #plot the boxplot but with the data after it was saved
    def plotEvolutionBoxplotSavedData(data_generations):
        fig, ax = plt.subplots()
        ax.boxplot(data_generations)
        
        ax.set(xlabel='generation', ylabel='Fitness - AUC',
        title='Boxplot Individual Fitness distribution throughout Generations')
        ax.grid()
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    
    # plots the evolution as a boxplot, to verify the individuals fitness distribution
    # throughout the evolution     
    def plotEvolutionBoxplot(self):
        fitness_generations=self.getFitnessIndividualsAsNumpyMatrix()
        fig, ax = plt.subplots()
        ax.boxplot(fitness_generations)
        
        ax.set(xlabel='generation', ylabel='Fitness - AUC',
        title='Boxplot Individual Fitness distribution throughout Generations')
        ax.grid()
        fig_to_be_handled=plt.gcf()
        
        return fig_to_be_handled
    
    # instead of list of fitnesses, returns the fitness evolution generations
    # as a 2D Numpy matrix, where each column represents the fitness individuals
    # values of a single generation, all columns represent all generation evolution
    def getFitnessIndividualsAsNumpyMatrix(self):
        fitness_generations=np.ones([self.population_size,
                                     len(self.individuals_fitness)])
        for i in range(0,len(self.individuals_fitness)):
            fitness_generations[:,i]=np.transpose(self.individuals_fitness[i])
        
        return fitness_generations
        
    # not used on the moment, but it will be done
    # for multiobjective for SMS-EMOA due do hypervolume
    # the goal was to have a 2-multiobjective problem
    # plot as red the non-dominated individuals and as
    # green the individual ones    
#    def plotMultiObjective(self):
#        hv=hypervolume(self.objective_coordinates)
#        ref_point=[1.1,1.1]
#        contributions=hv.contributions(ref_point) 
#        
#        fig, ax = plt.subplots()
#        colors=[]
#        labels=[]
#        plt.figure()
#        x=np.array(self.objective_coordinates[:,0])
#        y=np.array(self.objective_coordinates[:,1])
#        for i in range(0,len(self.objective_coordinates)):
#            if contributions[i]==0:
#                colors.append('green')
#                labels.append('dominated')
#            else:
#                colors.append('red')
#                labels.append('non_dominated')
#       
#        colors=np.array(colors)
#        labels=np.array(colors)
#        for color in ['green','red']:
#            loc=np.where(colors==color)[0]
#            print(colors)
#            print(color)
#            print(loc)
#            x_color=x[loc]
#            y_color=y[loc]
#            label_color=np.unique(labels[loc])
#            ax.scatter(x_color, y_color, c=color,label=label_color,
#                       alpha=0.3, edgecolors='none')
#            if color=='red':
#                arg_sort=np.argsort(x_color)
#                plt.plot(x_color[arg_sort],y_color[arg_sort])
#                    
#        ax.legend()
#        plt.title('Multiobjective Plot')
#        plt.xlabel('False Rate Positive')
#        plt.ylabel('1-Sensitivity')
#        ax.grid(True)
#        
#        plt.show()
#        
#        return colors

    
    # performs a mutation with a provided mutation_rate where
    # all individuals are iterated and mutated with the provided
    # mutation rate
    # the children obtained with mutation will be returned, in equal number
    # to the number of existing parents in the population    
    def mutateAllParentsUntilEqualPopulationNumber(self,mutation_rate):
        children_population=[]
        while len(children_population)<len(self.individuals):
            for individual in self.individuals:
                if random.random() < mutation_rate:
                    child=individual.mutate()
                    children_population.append(child) 
        return children_population
    
    def mutateParentsUntilEqualPopulationNumber(self,mutation_rate):
        children_population=[]
        #mating_pool_size=round(self.population_size/2)
        
        #tournament_size=2
        #mating_pool=self.tournamentSelection(tournament_size,mating_pool_size)
        mating_pool=self.TournamentSelection2on2()        
        while len(children_population)<self.population_size:     
            parent = self.selectParentFromMatingPool(mating_pool)
            if random.random()<mutation_rate:    
                child=Individual.mutate(parent)
                children_population.append(child)
            else:
                children_population.append(parent)
            
        return children_population
        
    
    
    # performs a mutation with a provided mutation_rate to a children population
    # all individuals are iterated and mutated with the provided
    # mutation rate
    # the children obtained with mutation will be returned
    def mutateAllChildren(mutation_rate,children):
        children_population=[]
        while len(children_population)<len(children):
            for individual in children:
                if random.random() < mutation_rate:
                    child=individual.mutate()
                    children_population.append(child) 
                else:
                    children_population.append(individual)
        return children_population  
    
    
    # sorts the population by a descending order in terms of fitness
    def sortPopulationByDescendingFitness(self):
        sorted_population=[]
        ascending_order=np.argsort(self.getIndividualsFitness(),axis=0)
        
        for index in reversed(ascending_order):
            sorted_population.append(self.individuals[int(index)])
        
        self.individuals=sorted_population
        
    # calculates the accumulated fitness and normalized so that the last element
    # is valued 1
    def calculateAccumulatedFitnessNormalized(self):
        individuals_fitness=self.getIndividualsFitness()
        individuals_fitness=individuals_fitness+abs(np.min(individuals_fitness))
        
        return (np.cumsum(individuals_fitness)/
                np.sum(individuals_fitness))
        
    
    # applies the stochastic universal sampling to select parents indexes to
    # then reproduce
    # the population is sorted descendently by its fitness
    # the accumulated fitness is calculated and regions of probability are
    # calculated
    # a random point is calculated and a series of pointers are obtained, where
    # all points are equidistant
    def stochasticUniversalSampling(self, step_pointer):
        self.sortPopulationByDescendingFitness()
        # to avoid when all individuals have fitness 0
        if np.sum(self.getIndividualsFitness())==0:
           indexes=[]
           for i in range (0,1):
               indexes.append(random.randint(0,len(self.individuals)-1))
               return indexes
        else:    
            pointers=Population.getEquallySpacedPointers(random.random(),step_pointer) 
            return self.selectIndividualsIndexByPointer(pointers)
    
    
    # for the vector of pointers, selects the pointer
    def selectIndividualsIndexByPointer(self,pointers):
        indexes=[]
        for pointer in pointers:
            if pointer < self.calculateAccumulatedFitnessNormalized()[0]:
                indexes.append(0)
            else:
                indexes.append(np.where(pointer>=self.calculateAccumulatedFitnessNormalized())[0][-1]+1)
        
        return indexes
    
    
    # giving a pointer, this method will return a series of pointers equally
    # space with the step_size of the pointer    
    def getEquallySpacedPointers(pointer_father, step_pointer):
        upper_part=np.arange(pointer_father,1,step_pointer)
        down_part=np.flip(np.arange(pointer_father,0,-step_pointer))
        return np.concatenate((down_part,upper_part[1:]),axis=0)
                
    
    # performs tournament selection
    #
    # - sorts descendently the population by its fitness
    # - chooses randomnly K participants to participate (tournament_size)
    # - chooses the one with higher fitness == the mininmum index since the population
    # was initially sorted descendently by its fitness
    # - if parent is not already in the mating pool, is added
    #
    # the tournament is performed until the size of the mating pool is filled    
    def tournamentSelection(self,tournament_size,mating_pool_size):
        self.sortPopulationByDescendingFitness()
        parents=[]    
        while (len(parents) < mating_pool_size):
            tournament_participants = np.random.choice(np.arange(0,self.population_size),
                                          tournament_size, replace = False)
            tournament_winner=np.min(tournament_participants)          
            parents.append(tournament_winner)
        return parents
    
    
    def TournamentSelection2on2(self):
        random.shuffle(self.individuals)
        parents=[]
        for i in range(0,len(self.individuals)-1):
            if self.individuals[i].fitness >= self.individuals[i+1].fitness:
                parents.append(i)
            else:
                parents.append(i+1)
                
        if self.individuals[len(self.individuals)-1].fitness>=self.individuals[0].fitness:
            parents.append(len(self.individuals)-1)
        else:
            parents.append(0)
        return parents
            

    def selectParentFromMatingPool(self,mating_pool):
        parent=np.random.choice(mating_pool)
        return self.individuals[parent]

          
    # this function returns two parents randomnly for mating    
    def selectParentsFromMatingPool(self, mating_pool):
        parent_1=np.random.choice(mating_pool)
        parent_2=np.random.choice(mating_pool)
        count=0
        while parent_1==parent_2:
            count=count+1
            parent_2=np.random.choice(mating_pool)
            # to avoid when i've only one parent or two that are different in the mating pool
            if count>10:
                parent_2=random.randint(0,len(self.individuals)-1)
            
        return self.individuals[parent_1], self.individuals[parent_2]
    
        
    # the population will suffer a parent selection through the chosen selection
    # method, "sus" (stochastic universal sampling) or "tournament" (tournament
    # selection) to create a mating pool
    # 
    # then, children will be born through the recombination of parents, randomnly
    # selected from the mating pool. these will born until reaching an equal number
    # to the population size
    def recombinateParentsUntilEqualPopulationNumber(self, method, mutation_rate,
                                                     recombination_rate):
        children_population=[]
        mating_pool_size=round(self.population_size/2)
        
        # Stochastic Universal Sampling
        if method == "sus":
            step=1/mating_pool_size
            mating_pool=self.stochasticUniversalSampling(step)
          
        #Tournament Selection    
        elif method == "tournament":
            #mating_pool=self.tournamentSelection(tournament_size,mating_pool_size)
            mating_pool=self.TournamentSelection2on2()
            #random.shuffle(mating_pool)
        recombination_rate=0.40
        random.shuffle(mating_pool)
        for i in range(0,len(mating_pool)-1):
            parent_1=mating_pool[i]
            parent_2=mating_pool[i+1]
            if random.random()<recombination_rate:
                child=Individual.recombination(self.individuals[parent_1],
                                               self.individuals[parent_2])
            else:
                child=self.individuals[parent_1]
        
            children_population.append(child)
            
        parent_1=mating_pool[len(mating_pool)-1]
        parent_2=mating_pool[0]
        if random.random()<recombination_rate:
                child=Individual.recombination(self.individuals[parent_1],
                                               self.individuals[parent_2])
        else:
            child=self.individuals[parent_1]
        children_population.append(child)    
        return children_population
                
    
    
    # evaluates the new born generation
    # decodes the phenotype of each child, evaluates the fitness and saves it
    # in the child information
    def EvaluateChildrenFitness(self,children):
        evaluated_children=[]
        for child in children:
            filter=Filter.createFilter(child.getDecodedPhenotype())
            if child.fitness==0:
               child.updateFitness(self.database.applyFilterToDatabaseWithPast(filter,
                                                                    "validation_data",
                                                                    "fitness","mean",0))
            evaluated_children.append(child)
        return evaluated_children
    
    # returns a boolean of fitness comparison where is true if fitness_1 is
    # bigger than fitness_2
    def isIndividualBetter(fitness_1,fitness_2):
        return fitness_1>fitness_2
    
    # removes from the population the provided individual    
    def removeIndividual(self,individual):
        self.individuals.remove(individual)
        
    # adds from the population the provided individual      
    def addIndividual(self, individual):
        self.individuals.append(individual)
    
    # retrieves the individual with the weakest fitness
    # iterates all individuals, finds the onde with the lowest fitness value
    def getWeakestIndividual(self):
        worst_fitness=1
        worst_individual = self.individuals[0]
        for individual in self.individuals:
            if individual.fitness<worst_fitness:
                worst_fitness=individual.fitness
                worst_individual=individual
        return worst_individual
    
    
    # performs an elitist environmental selection,
    # that is, from the children born, the ones better than the existing
    # parents replace them. thus, the parents with better fitness than the
    # children remain in the population. the others die
    def elitistEnvironmentalSelection(self, children):
        for child in children:
            weakest_parent=self.getWeakestIndividual()
            if Population.isIndividualBetter(child.fitness, weakest_parent.fitness):
                self.removeIndividual(weakest_parent)
                self.addIndividual(child)
                
    def generationSelection(self,children):
        self.individuals=children
                
    # saves the plot of evolution plotted with plotEvolutionMaxMeanMin with
    # the provided name
    def saveMaxMeanMinPlotEvolution(self,name_of_saved_file):
        plt=self.plotEvolutionMaxMeanMin()
        plt.savefig(name_of_saved_file + '_min_max_mean.pdf',edgecolor='black', dpi=400,)
        print("")
        print("Saving final plot...")
        
        
    # saves the plot of evolution plotted with plotEvolutionBoxplot with
    # the provided name
    def saveBoxplotEvolution(self,name_of_saved_file):
        plt=self.plotEvolutionBoxplot()
        plt.savefig(name_of_saved_file + '_boxplot.pdf',edgecolor='black', dpi=400,)
        print("")
        print("Saving final plot...")
    
    # retrieve the preprocessing labels to decode the electrode and characteristic
    # of the phenotype of each filter component
    def getPreprocessingLabels(self):
         os.chdir(self.path)
         return(open("preprocessing_labels.txt", "r").readlines())
         
    # for a filter component, change the filter component from the index of 
    # preprocessing label to the electrode and characteristic
    def changeIndexPreprocessingLabelsToRealName(self, component):
        component_index=int(component.split("__")[0])
        component_name=self.getPreprocessingLabels()[component_index]
        return (component_name.replace("\n","")+"__"+component.split("__",1)[1:][0])
    
    # for a filter phenotypical component, change from the electrode and characteristic
    # (wave or not) to the preprocessing label
    def changeRealNamesToIndexProcessingLabels(self,component):
        component_label=component.split("__")[0]
        component_index=self.getPreprocessingLabels().index(component_label+"\n")
        return (str(component_index)+"__"+component.split("__",1)[1:][0])
    
    def getCurrentFilterFromLoadedPhenotype(self,filter):
        for i in range(2,len(filter)):
            filter[i]=self.changeRealNamesToIndexProcessingLabels(filter[i])
        return filter
    
    def getAllCurrentFiltersFromLoadedFile(self,all_filters):
        coded_filters=[]
        for filter in all_filters:
            coded_filters.append(self.getCurrentFilterFromLoadedPhenotype(filter))
        return coded_filters
    
    # retrieves the filters with the electrode name and characteristic for all
    # the filters, aka all the population
    def getPhenotypeCurrentGeneration(self):
        phenotypes=[]
        for individual in self.individuals:
            filter=Filter.createFilter(individual.getDecodedPhenotype())
            for i in range(2,len(filter)):
                filter[i]=self.changeIndexPreprocessingLabelsToRealName(filter[i])
            phenotypes.append(filter)
            
        return phenotypes   
    
    # returns all the filters for the current generation
    def getCurrentIndividualFilters(self):
        filters=[]
        for individual in self.individuals:
            filters.append(Filter.createFilter(individual.getDecodedPhenotype()))
        