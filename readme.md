**REQUIREMENTS**

1. Install Python (if not present already)
	- Currently using Python 3.10.6
	- Also tested on 3.8.3 & 3.9.10

2. Install libraries needed:  (conda instructions)
	- numpy  ( conda install -c anaconda numpy )
	- scipy  ( conda install -c anaconda scipy )
	- scikit_learn ( conda install -c anaconda scikit-learn )
	- pandas ( conda install -c anaconda pandas ) (version < 2.0)
	- torch ( conda install -c pytorch pytorch )
	- psycopg2  ( conda install -c anaconda psycopg2 )
	- matplotlib ( conda install -c conda-forge matplotlib )
	- chart_studio ( conda install -c plotly chart-studio )
	- tqdm ( conda install -c conda-forge tqdm )
	- jinja2 ( conda install -c anaconda jinja2 )
	- pympler (for testing) ( conda install -c conda-forge pympler ) 
	- memory_profiler ( conda install -c conda-forge memory_profiler ) 
	- pybaseball ( pip install pybaseball )
	- multiprocess	( conda install multiprocess )
 	- openpyxl ( pip install openpyxl )

3. Clone "skill-estimation-framework" repo on Github

4. Navigate to the "skill-estimation-framework" directory

---

**TO RUN EXPERIMENTS**

1. Execute the following command to run experiments:

	```python runExp.py -domain specifyDomainHere -resultsFolder specifyFolderNameHere```

	**Parameters:**

		- domain: Specify which domain to use (1d, 2d, sequentialDarts, billiards, baseball)
		- delta:  Specify which resolution to use (for domain)

		For baseball domain specifically:
		- ids: List of pitcher IDs to use (Default: [])
		- types: List of pitch types (Defaults [])
		- startYear: Desired start year for the data (Default: 2021)
		- endYear: Desired end year for the data (Default: 2021)
		- startMonth: Desired start month for the data (Default: 1)
		- endMonth: Desired end month for the data (Default: 12)
		- startDay: Desired start day for the data (Default: 1)
		- endDay: Desired end day for the data (Default: 31)
		- every: After how many observations do a given experiment create checkpoints and reset info? (Default: 20)
		- savePlots: Flag to enable creating & saving the plots for the strikezone boards (both raw utility & EVsPerXskill ones) (Default: False)


	> NOTE: Control other parameters (number of experiments to perform, number of observations, which estimators to use,
	which agents to use, number of hypotheses, among others) inside *runExp.py*

	> NOTE: Control which agents to use and their respective parameters inside the *makeAgents.py* file of the respective domain.

	**Examples:**

	```python runExp.py -domain 1d -resultsFolder Testing1D```

	```python runExp.py -domain baseball -ids 642232 621237 -types FF```


---

**ADDITIONAL EXPERIMENTS**

The repository also includes a suite of scripts that measure how the JEEDS estimator reacts to suboptimal aiming in the darts domain. They support sharded local runs, Slurm submissions, and a verification harness for the parallel path. See [Testing/jeeds_aiming_sensitivity.md](Testing/jeeds_aiming_sensitivity.md) for a detailed overview of the workflow and usage examples.

---

**TO PROCESS RESULTS**

1. Execute the following command to process results once experiments are done:

	```python Processing/processResultsDarts.py -domain 1d -resultsFolder Experiments/1d/Results/```

	```python Processing/processResultsBaseball.py -resultsFolder Experiments/baseball/Testing/```

---
