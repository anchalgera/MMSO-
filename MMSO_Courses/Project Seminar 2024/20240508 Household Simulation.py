# Household Infection Simulation Code
# Generating households of different sizes and infection distributions per size
# The distributions change over discrete time steps according to a markov processes.

import numpy as np                              # Importing common Math Commands
import matplotlib.pyplot as plt                 # Plotting Graphs
from matplotlib.animation import FuncAnimation  # Animating Graphs and Plots

# -------------------------------------------------------------------------------------
# Auxiliary functions -----------------------------------------------------------------
# -------------------------------------------------------------------------------------

def factorial(n):
    '''
    Returns the factorial of a number, n! = n * (n-1) * (n-2) * ... * 3 * 2 * 1
    n: integer number
    '''
    res = 1
    while n > 1:
        res *= n
        n -= 1
    return res

def binomial_coefficient(n,k):
    '''
    Combination Function ('n choose k'): 
    Computes the number of ways to choose k elements from a set of 
    n elements without regarding the order of selection.
    n: elements available
    k: elements chosen
    '''
    if k > n:
        return 0
    return factorial(n) / (factorial(n-k) * factorial(k))

def print_matrix_rounded(M):
    '''
    Prints a matrix with rounded entries for better visibility.
    M: Matrix
    '''
    for row in M:
        for element in row:
            print('{:.3f}'.format(element), end='\t')
        print()

def print_constellations(AmountsVector):
    '''
    Prints details about the household distribution in the simulation.
    AmountsVector: List of [householdsize, [amounts with 0,1,2,... infections]].
    '''
    for entry in AmountsVector:
        size = entry[0]
        infections = entry[1]
        for i, amt in enumerate(infections):
            print(f'There are {amt} households of size {size}', end='') 
            print(f' with {i} infections.')

# -------------------------------------------------------------------------------------
# Infection Probabilities -------------------------------------------------------------
# -------------------------------------------------------------------------------------

def P(i,I,beta):
    ''' 
    Probability of getting infected from Household and outside ('globally').
    Calculates the probability that an individual becomes infected given i already 
    infected individuals in the Household and a global infection rate I.
    i: i infected individuals in a Household
    I: Infected population percentage
    beta: Infection rate
    '''
    return 1 - (1-beta)**i * (1-beta*I)

def q(n,j,i,I,beta):
    '''
    Probability of j agents getting newly infected in a Household of n people
    where i people are already infected.
    n: amount of people in the Household
    j: new infected agents in a Household
    i: already infected agents in a Household
    I: Percentage of globally infected population - used in P function
    beta: Infection rate - used in P function
    '''
    return binomial_coefficient(n-i,j) * P(i,I,beta)**j * (1 - P(i,I,beta))**((n-i)-j)

def r(k,i,gamma):
    '''
    Probability of k agents recovering in a given time step.
    k: agents recovering in the Household
    i: already infected agents in a Household
    gamma: Recovery rate
    '''
    return binomial_coefficient(i,k) * gamma**k * (1-gamma)**(i-k)

# -------------------------------------------------------------------------------------
# Markov Processes: Stochastic Transition Matrices ------------------------------------
# -------------------------------------------------------------------------------------

def Hij_transition_rate(n,i,j,k,I,beta,gamma):
    '''
    Calculating the values of the cells of the Markov Matrix H as
    transition probabilities from state H_i^n to H_(i+j-k)^n. Multiplies the 
    probabilities of j new infections and k recoveries to compute the transition rate 
    from i infected individuals to i + j - k individuals.
    n: amount of people in the Household
    i: already infected
    j: newly infected
    k: recovered
    I: Percentage of globally infected (used in P function called in q function)
    beta: Infection rate - used in P function which is called in q function
    gamma: Recovery rate - used in r function
    '''
    return q(n,j,i,I,beta) * r(k,i,gamma)
    
def H_markov_matrix(n,I,beta,gamma):
    '''
    Fills a Markov Matrix of all transition rates H_i^n to H_(i+j-k)^n, i = 0,...,n. 
    Creates a matrix to represent all possible transitions between states of infection 
    within the Household ensuring each row sums to 1 for probability conservation.
    n: Amount of people in the Household.
    I: Percentage of globally infected population
    beta: Infection rate
    gamma: Recovery rate
    '''
    H = np.zeros((n+1,n+1))
    for row in range(n+1):
        for col in range(n+1):
            # already infected i = row index since index starts at 0 infected
            i = row
            # new infections j = 0 if less or equal diagonal index, then counting up
            j = 0 if row >= col else col - i
            # recoveries k = counting downward, then 0 if at or higher diagonal index
            k = 0 if col >= row else row - col
            H[row,col] = Hij_transition_rate(n,i,j,k,I,beta,gamma)
    # Normalizing rows to sum up to 1 s.t. the resulting matrix is a stochastic matrix
    for row in range(n+1):
        total = np.sum(H[row, :])
        H[row, :] = H[row, :] / total
    return H

def Ht_Transition_Step(H,s_init,T):
    '''
    Calculate the household infection distribution for the next time step.
    H: The Household transition matrix
    s_init: initial distribution of infected people in a Household.
    T: Number of transitions, usually T=1 for one time update. 
       Bigger values of T would assume I to be constant.
    '''
    Ht = H
    if T > 1:
        Ht = np.linalg.matrix_power(H, T)
    # Row stochastic matrices only conserve vector length if multiplied from the right 
    # so the calculation is s_init^T * Hn 
    return np.dot(s_init, Ht)
    
# -------------------------------------------------------------------------------------
# Household Simulation ----------------------------------------------------------------
# -------------------------------------------------------------------------------------

def get_TotalNumberOfPopulation(CurrentAmounts):
    '''
    Returns the total number of people in the simulation from an AmountsVector entry
    CurrentAmounts: Vector storing one entry of the AmountsVector, of the form
    [Householdsize, [how many households of that size with 0,1,... infections]]
    '''
    Ntotal = 0
    for householdtype in CurrentAmounts:
        for amounts in householdtype[1]:
            Ntotal += householdtype[0] * amounts
    return Ntotal

def get_TotalNumberOfInfected(CurrentAmounts):
    '''
    Returns the current number of infected people from an AmountsVector entry
    CurrentAmounts: Vector storing one entry of the AmountsVector
    '''
    Itotal = 0
    for householdtype in CurrentAmounts:
        for i in range(len(householdtype[1])):
            householdsize = householdtype[0]
            householdinfection = householdtype[1][i]
            Itotal += i * householdinfection
    return Itotal

def get_InfectionVectors(CurrentAmounts):
    '''
    Retrieve the Infection Vectors (every second entry of the CurrentAmount vector). 
    Example:
    CurrentAmounts = [[1, [       10, 1]],  # retrieves [10, 1]
                      [2, [    5,  2, 1]],  # retrieves [5,2,1] and so on
    CurrentAmounts: Vector storing one entry of the AmountsVector
    '''
    HouseholdStateVectors = []
    for householdtype in CurrentAmounts:
        HouseholdStateVectors.append(householdtype[1])
    return HouseholdStateVectors

def GenerateTransitionMatrices(CurrentAmounts,IPercentage,beta,gamma):
    '''
    Generates Household Transition Matrices for each household size
    CurrentAmounts: Vector storing one entry of the AmountsVector
    IPercentage: Percentage of Infected People in the simulation
    beta: Infection rate constant
    gamma: Recovery rate constant
    '''
    HTransitionMatrices = [] 
    for i in range(len(CurrentAmounts)):
        householdsize = CurrentAmounts[i][0]
        HTransitionMatrix = H_markov_matrix(householdsize,IPercentage,beta,gamma)
        HTransitionMatrices.append(HTransitionMatrix)
    return HTransitionMatrices

def Household_Simulation(Timesteps,beta_infectionrate,gamma_recoveryrate,
                         InitialAmounts,printprogress=False):
    '''
    The household infection simulation.
    Timesteps: Amount of time steps in the simulation.
    beta_infection rate: Infection Rate in [0,1].
    gamma_recoveryrate: Recovery Rate in [0,1].
    InitialAmounts: Initial Households/Infections Distribution.
    printprogress: If true it will print the progress percentage of the computation.
    '''
    # Households of size n and their infection distribution over time
    # [amount of people n in the household, [amount of households of that size 
    #                                                       with 0,1,...n infections]]
    AmountsVector = []
    # Store different Household sizes separately, necessary for updating amounts later
    Householdsizes = [InitialAmounts[i][0] for i in range(len(InitialAmounts))]
    AmountsVector.append(InitialAmounts)
    # Count amount of people in the simulation from the newest (=[-1]-th) update
    Ntotal = get_TotalNumberOfPopulation(AmountsVector[-1])
    # Initializing a vector storing the total amount of currently infected people
    ItotalVector = []
    for t in range(Timesteps):
        # Update the current amount of infected people 
        ItotalVector.append(get_TotalNumberOfInfected(AmountsVector[-1]))
        # Generate the transition matrix using the global current Infection percentage 
        Current_H_Matrices = GenerateTransitionMatrices(AmountsVector[-1],
                                                        (ItotalVector[-1] / Ntotal),
                                                        beta_infectionrate,
                                                        gamma_recoveryrate) 
        # retrieve all infection distributions per household size
        Current_InfectionVectors = get_InfectionVectors(AmountsVector[-1])
        # Calculate the next household distributions by multiplying them
        Next_Transition = [Ht_Transition_Step(Current_H_Matrices[i],
                                              Current_InfectionVectors[i],1) 
                           for i in range(len(Current_H_Matrices))]
        # Update the AmountsVector with these new distributions
        NextAmounts = []
        for i in range(len(Householdsizes)):
            NextAmounts.append([Householdsizes[i], Next_Transition[i]])
        AmountsVector.append(NextAmounts)
        if printprogress:
            print(f'Progress: {t} / {Timesteps}')
    return AmountsVector, Ntotal, ItotalVector

# -------------------------------------------------------------------------------------
# Plotting functions ------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def plotItotalDevelopment(Ntotal, ItotalVector, beta, gamma):
    '''
    Plotting function for the total amount of infections of the simulation.
    Ntotal: Population size of the simulation.
    ItotalVector: Total amount of infections over all households over time.
    '''
    timevector = [i for i in range(len(ItotalVector))]
    labeltext = f'Infections for beta = {beta}, gamma = {gamma}'
    plt.plot(timevector, ItotalVector, label = labeltext)
    plt.title(f'Infections over time of a population of n = {Ntotal}')
    plt.xlabel('Simulation Time Increment')
    plt.ylabel('Total Amount of Infections')
    plt.legend()
    plt.show()

def plot_distributions_single(Curr_Amt):
    '''
    Plot a single time instance of the infection distributions for all household sizes.
    Curr_Amt: One entry of the AmountsVector, the 'current' state of all households
    '''
    num_plots = len(Curr_Amt)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))
    for i, vector in enumerate(Curr_Amt):
        title = vector[0]
        values = vector[1]
        positions = range(len(values))
        axs[i].bar(positions, values)
        axs[i].set_title(title)
        axs[i].set_xlabel('Position')
        axs[i].set_ylabel('Value')
    plt.tight_layout()
    plt.show()

def animate_distributions(vector_of_vector_lists,beta,gamma,repeatval=False):
    '''
    Plot a single time instance of all household sizes w. their infection distribution.
    vector_of_vector_lists: The AmountsVector.
    beta: infection rate
    gamma: recovery rate
    repeatval: True or False, to repeat the animation endlessly if True.
    '''
    fig, axs = plt.subplots(1, len(vector_of_vector_lists[0]), 
                            figsize=(len(vector_of_vector_lists[0]) * 5, 5))
    def update(frame):
        for i, vector in enumerate(vector_of_vector_lists[frame]):
            axs[i].clear()
            title = vector[0]
            values = vector[1]
            positions = range(len(values))
            axs[i].bar(positions, values)
            axs[i].set_title(title)
            axs[i].set_xlabel(f'with 0 to {vector[0]} infections')
            axs[i].set_ylabel(f'Amount of Households of size {vector[0]}')
        length = len(vector_of_vector_lists)-1
        titlestring1 = f'Household Distribution at time step: {frame}/{length}'
        titlestring2 = f' where beta = {beta}, gamma = {gamma}'
        titlestring  = titlestring1 + titlestring2
        fig.suptitle(titlestring, fontsize=16)
        plt.tight_layout()
        # If repeat is set to true, the code must be manually terminated
    if repeatval:
        print('The animation repeats endlessly. Manually close terminal to restart.')
    ani = FuncAnimation(fig, update, frames=len(vector_of_vector_lists), interval=0.1,
                        repeat=repeatval)
    plt.show()

# -------------------------------------------------------------------------------------
# Input / Output ----------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def example_smallhouseholds():
    '''
    Example for the household infection simulation with many small households.
    '''
    Timesteps          = 100
    beta_infectionrate = 0.2
    gamma_recoveryrate = 0.1
    # The initial Household Distribution
    # amount of infections increases from left to right by 1.
    # The following example reads as:
    # 10 households of size 1 with 0 infections, 1 household of size 1 with 1 infection
    #  5 households of size 2 with 0 infections, 2 with 1, 1 with 3
    #  5 households of size 3 with 0 infections, 1 with 1, 0 with 2 and 1 with 3.
    InitialAmounts = [[1, [10, 1      ]],  
                      [2, [ 5, 2, 1   ]],  
                      [3, [ 5, 1, 0, 1]] ] 
    AmountsVector, Ntotal, ItotalVector = Household_Simulation(Timesteps,
                                                           beta_infectionrate,
                                                           gamma_recoveryrate,
                                                           InitialAmounts)
    print('Initial Distribution: ')
    print_constellations(InitialAmounts)
    print('Final Distribution: ')
    print_constellations(AmountsVector[-1])
    # I over time
    plotItotalDevelopment(Ntotal, ItotalVector, beta_infectionrate,
                          gamma_recoveryrate)
    # Initial Distribution Histograms
    plot_distributions_single(AmountsVector[0])
    # Final Distribution Histograms
    plot_distributions_single(AmountsVector[-1])
    # Animation of all Histograms during the time steps of the simulation
    # animate_distributions(AmountsVector,beta_infectionrate,
    #                       gamma_recoveryrate,repeatval=True)

def example_largehousehold():
    '''
    Example for the household infection simulation with a large household.
    '''
    Timesteps          = 100
    beta_infectionrate = 0.2
    gamma_recoveryrate = 0.1
    vector100 = np.zeros(101)
    vector100[0] = 1
    InitialAmounts = [[  1, [0, 1]   ],
                      [100, vector100] ]
    AmountsVector, Ntotal, ItotalVector = Household_Simulation(Timesteps,
                                                           beta_infectionrate,
                                                           gamma_recoveryrate,
                                                           InitialAmounts,
                                                           printprogress=True)
    plotItotalDevelopment(Ntotal, ItotalVector, beta_infectionrate,
                          gamma_recoveryrate)
    
def example_realworlddata():
    '''
    Real world data for a household simulation (Germany, 2011)
    HOUSEHOLD DISTRIBUTION SOURCE:
    https://unstats.un.org/unsd/demographic/products/dyb/dyb_Household/4.pdf
    BETA, GAMMA SOURCE FOR COV19:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8993010/#pone.0265815.ref055
    '''
    Timesteps          = 100
    beta_infectionrate = 0.5
    gamma_recoveryrate = 0.13
    InitialAmounts = [[1, [13764950, 5               ]],  
                      [2, [12575548, 0, 0            ]],  
                      [3, [ 5184957, 0, 0, 0         ]],
                      [4, [ 3728176, 0, 0, 0, 0      ]],
                      [5, [ 1150307, 0, 0, 0, 0, 0   ]],
                      [6, [  529095, 0, 0, 0, 0, 0, 0]] ]
    AmountsVector, Ntotal, ItotalVector = Household_Simulation(Timesteps,
                                                           beta_infectionrate,
                                                           gamma_recoveryrate,
                                                           InitialAmounts)    
    plotItotalDevelopment(Ntotal, ItotalVector, beta_infectionrate,
                          gamma_recoveryrate)                  
    
# Evaluating the examples by calling their functions
#example_smallhouseholds()
#example_largehousehold()
example_realworlddata()