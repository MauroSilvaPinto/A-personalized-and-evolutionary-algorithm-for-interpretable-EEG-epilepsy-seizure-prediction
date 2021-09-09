"""
This is the script you will need to run to execute the EA.
The output of the EA will be saved in Data folder.


You can use the following EA configurations:
    # methods
        # A: elitist = mutation_rate_1.00 
        # B: elitist = mutation_rate_1.00 & recombination_rate_0.80 (the one in the paper)
        # C: gerational = mutation_rate_0.40
        # D: gerational = mutation_rate_0.40 $ recombination_rate 0.20
        # E: random_search

    To adjust the minimum pre-ictal period, you will need to go to the Feature
    class, and adjust the genotype of the pre-ictal range.
    
    For example, if you want a minimum SOP of 30 minutes, you will need to adjust
    this function as:
          def getPreIctalRange():
              return np.arange(30,50,5)
          
    Or if you want with a minimum SOP of 40 minutes: 
          def getPreIctalRange():
              return np.arange(40,60,5)
              
"""


# %% Imports

from timeit import default_timer as timer
from Database import Database
from Population import Population
from Filter import Filter
import warnings
import numpy as np
import os
import copy
import matplotlib as plt


    

# %% System settings

# clear the screen
os.system('clear')
# ignores warnings
warnings.filterwarnings("ignore")


# %% Loading the Database and Prunning it for the first time

# go back to data folder
os.chdir("..")
os.chdir("..")
os.chdir("Data")
path_data=os.getcwd()
os.chdir("Preprocessed_data")
path=os.getcwd()

# Save the run files in the data folder
path_stored_files=path_data
os.chdir(path)
filenames=os.listdir()       

print("Loading the database ...")

# Load Database
my_database=Database(path)

print("Eliminating seizures with less than 240 minutes...")
my_database.eliminateSeizuresWithLessThan(230)

print("Eliminating patients with less than 5 seizures ...")
my_database.eliminatePatientsWithLowSeizureNumber(5)
print(" ")



# Print the number of seizures for each patient
my_database.printSeizuresFromAllPatients()

# Make a list of all patients
my_database_patient_list=my_database.patient_list


# %%  Parameters of the population


for run in [1]:
    
    number_of_features=5
    population_size=100
    
    print("---------------------------------------------------")
    print("number of features: " + str(number_of_features))
    print("population size: " + str(population_size))
    print("run: "+str(run))
    print("---------------------------------------------------")
    print("")
    
    
    # %% Iterating all patients
    
    # iterating all patients
    for i in range(0,1):#len(my_database_patient_list)):
        
        # methods
        # A: elitist = mutation_rate_1.00 
        # B: elitist = mutation_rate_1.00 & recombination_rate_0.20
        # C: gerational = mutation_rate_0.40
        # D: gerational = mutation_rate_0.40 $ recombination_rate 0.20
        # E: random_search
        
        
        method="A"
        
        if method=="A":
            mutation_rate=1.00
            recombination_rate=0.00
            replacement="elitist"
        elif method=="B":
            mutation_rate=1.00
            recombination_rate=0.80
            replacement="elitist"
        elif method=="C":
            mutation_rate=0.40
            recombination_rate=0.00
            replacement="gerational"
        elif method=="D":
            mutation_rate=0.40
            recombination_rate=0.20
            replacement="gerational"
        elif method=="E":
            # so para a formula bater certo
            mutation_rate=1.00
            recombination_rate=0.00
            replacement="gerational"
            

        number_evaluations=15000
        iterations = round((number_evaluations/
                      (population_size *(recombination_rate+(1-recombination_rate)*mutation_rate))))
        
        
        print("Chosen method: "+method)
        print("Replacement: "+replacement)
        print("Mutation rate: "+str(mutation_rate))
        print("Recombination rate: "+str(recombination_rate))
        print("Iterations: "+str(iterations))
        print("---------------------------------------------------")
        print("")
        
        
        
        
        patient=int(my_database_patient_list[i])
        print("Training an EA Feature Selection for patient " + str(patient) + "...")
        print(" ")
        
        
        # Creating database for only the provided patient
        print("Creating database for patient " + str(patient) + "...")
        one_patient_database=copy.deepcopy(my_database)
        one_patient_database.eliminateAllPatientsExcept(patient)
        one_patient_database.eliminateSeizuresWithLessThan(230)
    
    
    
        # %% Initialization of the population
        
        print("Initializing the population ...")
        # Initialize the population
        my_population=Population(population_size,number_of_features,path,one_patient_database)
        print("Evaluating the population first elements ...")
        my_population.evaluatePopulationFitness()
        my_population.updateBestIndividual()
    
           
        # %% Performs the evolution
        
        
        print("Starting the population evolution ...")
        
        
        # or while a stoppage criteria
        for i in range(0,iterations):
            start=timer()
            
            # performing selection, recombination and mutation depending on the
            # chosen method
            
            # method A: elitist = mutation_rate_1.00 
            if method=="A":
                children_generation=my_population.mutateAllParentsUntilEqualPopulationNumber(mutation_rate)

            # method B: elitist = mutation_rate_1.00 & recombination_rate_0.20
            elif method=="B":
                children_generation=my_population.recombinateParentsUntilEqualPopulationNumber("tournament", mutation_rate,recombination_rate)
                children_generation=Population.mutateAllChildren(mutation_rate,children_generation)
                        
            # method C: gerational = mutation_rate_0.40               
            elif method=="C":
                children_generation=my_population.mutateParentsUntilEqualPopulationNumber(mutation_rate)
            
            # method D: gerational = mutation_rate_0.40 $ recombination_rate 0.20
            elif method=="D":
                children_generation=my_population.recombinateParentsUntilEqualPopulationNumber("tournament", mutation_rate,recombination_rate)
                children_generation=Population.mutateAllChildren(mutation_rate,children_generation)
            
            # method E: random_search
            elif method=="E":
                children_generation=my_population.generateRandomOffspring()
            
            # evaluate children's fitness
            children_generation=my_population.EvaluateChildrenFitness(children_generation)


            # environmental replacement
            if replacement=="elitist":
               my_population.elitistEnvironmentalSelection(children_generation)    
            elif replacement=="gerational":
               my_population.generationSelection(children_generation)  
           
            # update population fitness
            my_population.updateIndividualsFitness()
            
            # update fitness metrics of population
            my_population.updateGenerationFitnessMetrics()
            
            # update best individual
            my_population.updateBestIndividual()
                
            
            if np.max(my_population.individuals_fitness[-1])>0.99:
                print("Evolution stopped because one individual reached maximum fitness.")
                break
            
            
            # with elitistic only pleaseeeee
            if i > 50 and (method=="A" or method=="B") and (abs(np.max(my_population.individuals_fitness[-1]-
                                                 my_population.individuals_fitness[-50]))<0.01):
                print("Evolution stopped because no fitness improvement occurred for more than 50 generations")
                break
            
            print("------------------ Iteration " + str(i) +  (" ------------------"))
            end=timer()
            
            # prints the time one iteration took
            print(end-start)
            
        
        print("")
        print("The evolution process has ended.")
        print("")
        print("Saving files now ...")
        
            
        os.chdir(path_stored_files)   
        name_of_saved_file=("patient_"+str(patient)+
                            "_method_"+method+
                            "_population_"+str(population_size) +
                            "_mutation_"+str(mutation_rate)+
                            "_recombination_"+str(recombination_rate)+
                            "_run_" + str(run))
        #Saving Figures
        my_population.saveMaxMeanMinPlotEvolution(name_of_saved_file)
        my_population.saveBoxplotEvolution(name_of_saved_file)
        
        # Saving Population Characteristics
        
        print("Saving Individual Fitnesses Throughout Evolution ...")
        fitness_individuals=my_population.getFitnessIndividualsAsNumpyMatrix()
        os.chdir(path_stored_files)                   
        np.save((name_of_saved_file+"_fitness_individuals"),fitness_individuals)
        
        print("Saving Individual Current Phenotypes ...")
        phenotype_individuals=my_population.getPhenotypeCurrentGeneration()
        os.chdir(path_stored_files)                   
        np.save((name_of_saved_file+"_phenotype_individuals"),phenotype_individuals)
        
        print("Saving Best Individual Fitness...")
        best_fitness=my_population.best_solution_fitness
        os.chdir(path_stored_files)                   
        np.save((name_of_saved_file+"_best_fitness"),best_fitness)
        
        
        print("Saving Best Individual Phenotype...")
        best_individual=my_population.best_solution_individual
        best_filter=Filter.createFilter(best_individual.getDecodedPhenotype())
        for i in range(2,len(best_filter)):
            best_filter[i]=my_population.changeIndexPreprocessingLabelsToRealName(best_filter[i])
        os.chdir(path_stored_files)                   
        np.save((name_of_saved_file+"_best_individual"),best_filter)
        
        
    
    
    
