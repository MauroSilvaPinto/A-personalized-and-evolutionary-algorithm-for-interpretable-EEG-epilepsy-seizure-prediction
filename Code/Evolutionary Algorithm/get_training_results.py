'''
A code to get the training information from 
the evolutionary training execution, from a given patient

'''

# %% Imports

import os

from Database import Database
from Population import Population
import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
from Filter import Filter
import copy
import scipy
from scipy import stats


# %% System settings

# clear the screen
os.system('clear')
# ignores warnings
warnings.filterwarnings("ignore")


my_database_patient_list=['53402']

mean_seizure_sensibilities=[]
mean_fpr=[]
mean_ratio=[]
mean_fitnesses=[]
mean_sample_sensibilities=[]
mean_pre_ictal=[]
mean_number_seizures=[]

std_seizure_sensibilities=[]
std_fpr=[]
std_ratio=[]
std_fitnesses=[]
std_sample_sensibilities=[]
std_pre_ictal=[]
std_number_seizures=[]

for patients in range(0,len(my_database_patient_list)):
    

    # %% Loading the Database and Prunning it
    # go back to data folder
    os.chdir("..")
    os.chdir("..")
    os.chdir("Data")
    path_data=os.getcwd()
    
    path_stored_files=os.getcwd()+'/Evolutionary_executions/sop_min_40'
    os.chdir("Preprocessed_data")
    path=os.getcwd()
    
    
    os.chdir(path)
    filenames=os.listdir()    
    
    patient=my_database_patient_list[patients]
       
    # Load Database
    my_database=Database(path)
    
    
    my_database.eliminateSeizuresWithLessThan(230)
    my_database.eliminatePatientsWithLowSeizureNumber(5)

        
    my_database.eliminateAllPatientsExcept(int(patient))
    my_database.eliminateSeizuresWithLessThan(230)
    path_stored_files=(path_stored_files+"/patient_"+patient)
    os.chdir(path_stored_files)
    
    # %%  Parameters of the population
    number_of_features=1
    population_size=1
    
    # %% Initialization of the population
    
    # Initialize the population
    my_population=Population(population_size,number_of_features,path,my_database)
    
    # %% Loading the Database and Prunning it
    os.chdir(path_stored_files)
    
    # %% Analyzing Best Individual for Validation
    
    print("--------------------------------------")
    
    
    for forget_this in range(0,1):
                
        fitnesses=[]
        sensibilities=[]
        correct_seizures=[]
        pre_ictals=[]
        sensibilities=[]
        ratio_penalties=[]
        
        fps=[]
        fprs=[]
        false_alarms=[]
        
    
        for run in range(1,31):
            os.chdir(path_stored_files)
            
            population=100
            method="B"
            mutation="1.0"
            recombination="0.8"
            kind="one_by_one"
                
            
            run_1_filter=np.load('patient_'+str(patient)+'_method_'+
                     method+'_population_'+str(population)+
                     "_mutation_"+mutation+"_recombination_"+recombination+
                     '_run_' + str(run)+'_best_individual.npy')
            run_1_filter_phenotype=copy.deepcopy(run_1_filter)
            run_1_filter_phenotype=my_population.getCurrentFilterFromLoadedPhenotype(run_1_filter_phenotype)
            
            
            
            
            metrics=my_database.applyFilterToDatabaseWithPast(run_1_filter_phenotype,
                                                                                           "validation_data",
                                                                                           "sensibility_fpr_hour",
                                                                                           "all_patients",
                                                                                           0)
            
            fitnesses.append(metrics[0][6])
            sensibilities.append(metrics[0][2])
            correct_seizures.append(metrics[0][5])
            
            fprs.append(metrics[0][7])
            fps.append(metrics[0][3])
            false_alarms.append(metrics[0][4])
            pre_ictals.append(metrics[0][8])
            ratio_penalties.append(metrics[0][7])
            
        
        print("")
        print("Patient:"+patient)
        print("Fitness: "+str(round(np.mean(fitnesses),2))+
              " +/- "+str(round(np.std(fitnesses),2)))
        print("Sample Sensibility: "+str(round(np.mean(sensibilities),2))+
              " +/- "+str(round(np.std(sensibilities),2)))
        print("Ratio Penalty: "+str(round(np.mean(ratio_penalties),2))+
              " +/- "+str(round(np.std(ratio_penalties),2)))
        print("Seizure Sensibility: "+str(round(np.mean(correct_seizures)/metrics[0][1],2))+
              " +/- "+str(round(np.std(correct_seizures)/metrics[0][1],2)))        
        print("FPR/h: "+str(round(np.mean(fps),2))+
              " +/- "+str(round(np.std(fps),2)))
        print("Pre-Ictal Times:"+str(round(np.mean(pre_ictals),2))+
              " +/- "+str(round(np.std(pre_ictals),2)))
        print("Number of Seizures: "+str(metrics[0][1]))
        
       
        
        mean_seizure_sensibilities.append(np.mean(correct_seizures)/metrics[0][1])
        mean_fpr.append(np.mean(fps))
        mean_ratio.append(np.mean(ratio_penalties))
        mean_fitnesses.append(np.mean(fitnesses))
        mean_sample_sensibilities.append(np.mean(sensibilities))
        mean_pre_ictal.append(np.mean(pre_ictals))
        mean_number_seizures.append(np.mean(metrics[0][1]))
        
        
        std_seizure_sensibilities.append(np.std(correct_seizures)/metrics[0][1])
        std_fpr.append(np.std(fps))
        std_ratio.append(np.std(ratio_penalties))
        std_fitnesses.append(np.std(fitnesses))
        std_sample_sensibilities.append(np.std(sensibilities))
        std_pre_ictal.append(np.std(pre_ictals))
        std_number_seizures.append(np.std(metrics[0][1]))
        
print("")
print("")
print(" ----------------------------- Total Overview --------------------")

print("Fitness: "+str(round(np.mean(mean_fitnesses),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_fitnesses)**(2)/(19**2))),2)))

print("Sample Sensibility: "+str(round(np.mean(mean_sample_sensibilities),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_sample_sensibilities)**(2)/(19**2))),2)))

print("Ratio Penalty: "+str(round(np.mean(mean_ratio),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_ratio)**(2)/(19**2))),2)))

print("Seizure Sensibility: "+str(round(np.mean(mean_seizure_sensibilities),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_seizure_sensibilities)**(2)/(19**2))),2)))
    
print("FPR/h: "+str(round(np.mean(mean_fpr),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_fpr)**(2)/(19**2))),2)))

print("Pre-Ictal Times:"+str(round(np.mean(mean_pre_ictal),2))+
      " +/- "+str(round(np.sqrt(np.sum(np.array(std_pre_ictal)**(2)/(19**2))),2)))
