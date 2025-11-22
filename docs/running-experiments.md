# Running Experiments

Use `runExp.py` to launch experiments for different domains.

## Command
```bash
python runExp.py -domain <domain> -resultsFolder <folder_name>
```

### Key parameters
- `domain`: Domain to use (`1d`, `2d`, `sequentialDarts`, `billiards`, `baseball`).
- `delta`: Resolution parameter for the chosen domain.
- Baseball-specific options: `ids`, `types`, `startYear`, `endYear`, `startMonth`, `endMonth`, `startDay`, `endDay`, `every`, `savePlots`.

> Adjust additional settings (e.g., number of experiments, observations, estimators, agents, hypotheses) in `runExp.py`.
> Configure agent choices and parameters in the domain-specific `makeAgents.py` file.

### Examples
```bash
python runExp.py -domain 1d -resultsFolder Testing1D
python runExp.py -domain baseball -ids 642232 621237 -types FF
```
