import numpy as np
from scipy import stats
from numpy import random
import time
from copy import deepcopy
from pandas import DataFrame
import sys

# Increase python recursion limit so N=5000 does cause OOM error for quick sort worst case; beware if you're running hardware from before 2000 (if so, why?)
sys.setrecursionlimit(100000)

class Distribution:
    def __init__(self, n):
        self.data = None
        self.n = n
        self.sortedness = None
        self.skew = None
        self.cardinality = None
        self.cv = None
    
    #Need to randomly vary parameters that govern sortedness, skew, cardinality, and standard deviation, then use those to build the distribution.
    #Start by setting number of values, then range of values (governs cardinality/standard deviation), then exponentiate by some factor (handles skew),
    #Then sort the distribution and apply a random number of random swaps (presortedness). Then take measurements.

    """Measurements: 
        Sortedness [stats.kendalltau(data, sorted data)] --> Self explanatory
        Skew [stats.skew()] --> Some algorithms may struggle with skewed data
        Cardinality [Find num of repeat values from number of unique ones] --> Some algorithms may struggle with repeated values
        Standard deviation [use numpy] --> Some algorithms do better with less dense data"""

    def generate_data(self):
        """
        Use random multipliers for the randint() upper bound of the min and max of the range to ensure cardinality varies. If the multipliers aren't used, cardinality behaves similarly to the birthday problem.
        In other words, the number of unique values tends to a constant for a fixed range as n approaches the max of the range. If the range max is unchanged for all distributions with n elements, cardinalities will be similar.
        We want variation of cardinality between distributions of the same n, so the range's size, and thus the range max, must be randomized.
        """

        # Randomly choose distribution range, exponent factor, and number of swaps
        low = random.randint(0, int(self.n * (0.1 + (random.random() * 5))))
        high = random.randint(0, int(self.n * (0.1 + (random.random() * 5))))
        exponent_factor = 0.3 + (random.random() * 3)
        numswaps = random.randint(0, self.n)
        data = []

        #Edge-case handling: random.randint(low, high) requires high > low, and low < 0.
        if low > high:
            low, high = high, low
        
        elif low == high:
            high += 1
        
        
        # Populate data within the range
        for i in range(self.n):
            data.append(random.randint(low, high))
        
        # Exponentiate to introduce skew

        # NOTE: Every element after exponentiation is rounded to 5 decimal places. 
        # If this is not done, floating point error at the last few decimal places results in differing floats for identical integers, 
        # which would violate our treatment of cardinality as an independent variable.
        data = (np.array(data) ** exponent_factor).round(5).tolist() 

        # Execute a random number of random swaps after sorting
        data = np.sort(data).tolist()
        for i in range(numswaps):
            index_1 = random.randint(0, self.n - 1)
            index_2 = random.randint(0, self.n - 1)
            data[index_1], data[index_2] = data[index_2], data[index_1]
        
        self.data = data
        return
    
    def calc_data_params(self):
        self.sortedness = float(stats.kendalltau(self.data, np.sort(self.data).tolist())[0]) #-1 is least sorted, 1 is most sorted
        self.skew = float(stats.skew(self.data))
        self.cardinality = float((len(np.unique(self.data))) / self.n) #normalize by n
        self.cv = float(np.std(self.data) / np.mean(self.data)) #Use coefficient of variance: std / mean
        return


class Sort:
    def __init__(self):
        self.sort_time = None
        self.assigns = 0 # 2 assigns per swap, 1 otherwise (i.e. insertion sort's shifting)
        self.comps = 0
        self.algs = {self.bubble_sort : "Bubble Sort", self.selection_sort : "Selection Sort", self.insertion_sort : "Insertion Sort", self.merge_sort : "Merge Sort", self.quick_sort : "Quick Sort"}

    def reset(self):
        self.assigns = 0
        self.comps = 0
        self.sort_time = None
    
    def selection_sort(self, numList):
        size = len(numList)
        for pos in range(size):
            minpos = pos
            for seek in range(pos,size):    # find smallest in range
                self.comps += 1
                if numList[seek] < numList[minpos]:
                    minpos = seek

            numList[minpos],numList[pos] = numList[pos],numList[minpos]
            self.assigns += 2
        

    def insertion_sort(self, numList):
        
        size = len(numList)
        for pos in range(1,size):
            value = numList[pos]             
            self.assigns += 1
            pos_found = False
            
            while not pos_found and pos > 0:
                self.comps += 1
                if value < numList[pos - 1]:
                    self.assigns += 1
                    numList[pos] = numList[pos - 1]
                    pos -= 1
        
                else:
                    pos_found = True

            self.assigns += 1
            numList[pos] = value
    
    def bubble_sort(self, numList):
        size = len(numList)
        
        sorted = False                     
        while not sorted:
            sorted = True                   
            for pos in range(1,size):
                
                self.comps += 1
                if numList[pos] < numList[pos-1]:
                    numList[pos], numList[pos - 1] = numList[pos - 1], numList[pos]
                    self.assigns += 2
                    sorted = False
    
    def merge(self, lst1, lst2):
        merged = []
        i = 0
        j = 0

        while i < len(lst1) and j < len(lst2):
            self.comps += 1
            self.assigns += 1

            if lst1[i] < lst2[j]:
                merged.append(lst1[i])
                i += 1
            
            else:
                merged.append(lst2[j])
                j += 1
        
        if i < len(lst1):
            while i < len(lst1):
                self.assigns += 1
                merged.append(lst1[i])
                i += 1
        
        else:
            while j < len(lst2):
                self.assigns += 1
                merged.append(lst2[j])
                j += 1
        
        return merged
    
    def merge_sort(self, numList):
        if len(numList) == 1:
            return numList
        
        mid = len(numList) // 2

        left = self.merge_sort(numList[:mid])
        right = self.merge_sort(numList[mid:])
        numList[:] = self.merge(left, right) #We just want stats, no need to return the sorted array.
        return numList


    def partition(self, numList, first, last):
        pivot = numList[last]
        left = first
        right = last - 1

        done = False
        while not done:
            while numList[left] <= pivot and left <= right:
                self.comps += 1
                left += 1
            
            while numList[right] >= pivot and right >= left:
                self.comps += 1
                right -= 1
            
            if right < left:
                done = True
            
            else:
                self.assigns += 2
                numList[right], numList[left] = numList[left], numList[right]
        
        self.assigns += 2
        numList[last], numList[left] = numList[left], numList[last]
        return left
    
    def quick_sort(self, numList, start=0, end=None):
        if end is None:
            end = len(numList) - 1
        
        if len(numList) <= 1:
            return numList
        
        if start < end:
            split = self.partition(numList, start, end)
            self.quick_sort(numList, start, split - 1)
            self.quick_sort(numList, split + 1, end)
        return numList

    def run_sort(self, distribution, sort_func):
        self.reset() #Just in case
        
        nums = deepcopy(distribution.data)

        #Actually sort now
        start = time.perf_counter()
        sort_func(nums)
        end = time.perf_counter()

        self.sort_time = end - start
        return (distribution.n, self.algs[sort_func], distribution.cv, distribution.cardinality, distribution.skew, distribution.sortedness, self.sort_time, self.comps, self.assigns)

# DATA-PRODUCING/MANAGEMENT FUNCTIONS---------------------------------------------------------------------------

def perform_group_run(n, runs_per_alg):
    sorter = Sort()

    #algorithm functions to loop over (convenient use of that dictionary!)
    alg_funcs = list(sorter.algs.keys())

    run_stats = {"n" : [], "Algorithm" : [], "CV" : [], "Cardinality" : [], "Skew" : [], "Sortedness" : [], "Time" : [], "Comparisons" : [], "Assignments" : []}

    run_num = 1
    total_runs = len(alg_funcs) * runs_per_alg

    #This is genuinely horrendous, no clue how to streamline this... should I even try?

    #Loop over every algorithm, sorting a random distributions runs_per_alg times. Distribution parameters and sorting metrics get stored after each individual run
    for alg in alg_funcs:
        for i in range(runs_per_alg):
            print(f"Executing N={n} sorting run using {sorter.algs[alg]}: {run_num}/{total_runs}")

            # Generate distribution and parameters
            dist = Distribution(n)
            dist.generate_data()
            dist.calc_data_params()
            

            # generate_data() occasionally builds distributions where every value is the same. 
            # This breaks skew and sortness calculations via catastrophic cancellation. Floating point error strikes again...
            # These distributions have extremely small CVs, usually O(10^-16). If this is found, regenerate the distribution until the CV is acceptable.
            single_val_dist = True
            while single_val_dist:
                single_val_dist = False

                if dist.cv < 10e-14 or len(np.unique(dist.data)) == 1: #Set detection limit 100x larger than commonly observed CV for this case, CVs for valid distributions will never get this small anyway
                    single_val_dist = True
                    dist = Distribution(n)
                    dist.generate_data()
                    dist.calc_data_params()


            # Sort using the algorithm it's currently iterating over
            sort_stats = sorter.run_sort(dist, alg)

            #Put it into a dictionary with identical keys compared to the group run's dictionary
            data = {"n" : sort_stats[0], 
                    "Algorithm" : sort_stats[1], 
                    "CV" : sort_stats[2], 
                    "Cardinality" : sort_stats[3] , 
                    "Skew" : sort_stats[4], 
                    "Sortedness" : sort_stats[5], 
                    "Time" : sort_stats[6], 
                    "Comparisons" : sort_stats[7], 
                    "Assignments" : sort_stats[8]}
            
            #Store the individual run's data in the group run's dictionary
            for key, value in data.items():
                run_stats[key].append(value)
            
            run_num += 1

    return run_stats

def generate_data_file(data, n):
    df = DataFrame(data)

    #Save to .pkl format. Easily read by pandas and preserves datatypes (we can avoid needing to cast everything from strings when reading)
    df.to_pickle(f"sort_run_n{n}.pkl")

    #For viewing purposes
    df.to_csv(f"sort_run_n{n}.csv")
    return
    


def main():
    #Change this to add more runs or change the distribution size of existing ones. Each set of runs for a particular n is saved as its own file for later use.
    DIST_RUN_SIZES = [100, 1000, 5000]

    #Change this to add more data points per set of n-sized distributions.
    RUNS_PER_ALG = 100

    for size in DIST_RUN_SIZES:
        run_stats = perform_group_run(size, RUNS_PER_ALG)
        generate_data_file(run_stats, size)

if __name__ == "__main__":
    main()



        