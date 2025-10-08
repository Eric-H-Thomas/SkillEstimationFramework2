import argparse
import subprocess 
import time
import os 
import json

if __name__ == "__main__": 

    # Get arguments from command line
    parser = argparse.ArgumentParser(description='Make instructions to run missing experiments.')
    parser.add_argument('-file','-file', help='Name of the file that contains the IDs of the pitchers to use',default = "")
    args = parser.parse_args()

    if args.file == "":
        print('No Missing Experiment File Provided')
        exit()
    else:
        print('[MAIN] Missing Experiment file is: ', args.file)
        with open(args.file,"r") as inFile:
            fileContents = json.load(inFile)
            missing_experiments = fileContents['missingExps']

    spawn_list = []

    #Generate the list of all the spawn parameters
    for ex in missing_experiments:
        #Get the pitcherID and pitchType
        pitcherID = ex[0]
        pitchType = ex[1]

        #Create the argument list
        arguments = []
        arguments.append('python')
        arguments.append('runExpBaseball.py')
        # arguments.append('testspawn.py') #FOR TESTING
        # arguments.append('-domain')
        # arguments.append('baseball')
        arguments.append('-ids')
        arguments.append(f'{pitcherID}')
        arguments.append('-types')
        expFolder = f"ForJournal-Take2-Experiment-PitcherID-{pitcherID}-PitchType-{pitchType}"
        targ = arguments[:]
        targ.append(f'{pitchType}')
        targ.append('-resultsFolder')
        targ.append(f'{expFolder}')
    
        #Add this to our list of processes to spawn
        spawn_list.append(targ)

    print(f'[MAIN] Created a spawn list with {len(spawn_list)} processes in it.')

    #Now we spawn the processes from the list
    current_spawns = []
    N = 30
    spawn_index = 0
    finished_spawns = 0

    while finished_spawns < len(spawn_list):
        print('[MAIN] We have ', finished_spawns, 'finished, out of ', len(spawn_list))

        #Check on our spawns, to see if any have finished           
        for s in current_spawns: 
            if s.poll() is not None:
                #If it finished, then remove it
                current_spawns.remove(s)
                finished_spawns += 1
                print(f'[MAIN] Spawn finished. {len(current_spawns)} still running. {len(spawn_list) - spawn_index} left to spawn.')

        #Spawn enough to fill all our N slots, as long as we have things left to spawn
        while (len(current_spawns) < N) and (spawn_index < len(spawn_list)):
            #Get the next process to spawn
            next_spawn = spawn_list[spawn_index]
            spawn_index += 1
            print('[MAIN] Spawning process: ', next_spawn)
            spawn = subprocess.Popen(next_spawn)
            current_spawns.append(spawn)
            if spawn_index == len(spawn_list):
                print(f'[MAIN] Done spawning all {len(spawn_list)} subprocesses')

        #Rest until we check everything again
        print('[MAIN] Sleeping. . . .')
        time.sleep(1)

    print('[MAIN] All spawns finished.')