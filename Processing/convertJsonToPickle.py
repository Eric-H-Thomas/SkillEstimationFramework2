import argparse
import os
import code
import json, pickle
import time

def toPickle(file):

    t1 = time.perf_counter()

    # Read JSON file
    with open(file,'r') as json_file:
        data = json.load(json_file)
    
    # Write to pickle file
    with open(file,'wb') as pickle_file:
        pickle.dump(data,pickle_file)

    return time.perf_counter()-t1


if __name__ == "__main__":


    # Get arguments from command line
    parser = argparse.ArgumentParser(description='Processing results')
    parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")   
    args = parser.parse_args()


    # Prevent error with "/"
    if args.resultsFolder[-1] == os.sep:
        args.resultsFolder = args.resultsFolder[:-1]
    

    result_files = os.listdir(args.resultsFolder+os.sep+"results")


    if len(result_files) == 0:
        print("No result files present for experiment.")
        exit()


    try:
        result_files.remove("backup")
    except:
        pass

    try:
        result_files.remove(".DS_Store")
    except:
        pass



    times = []

    total_num_exps = 1
    for rf in result_files: 
        print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)
        totalTime = toPickle(args.resultsFolder+os.sep+"results"+os.sep+rf)
        times.append(totalTime)
        total_num_exps += 1

    print(f"AVG time per file: {sum(times)/len(times):.2f}")





