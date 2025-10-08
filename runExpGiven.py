import argparse, datetime, os, time, pickle, setupEnv
import numpy as np
from multiprocess import Process, Queue
from Estimators.estimators import Estimators as EstimatorsClass, ObservedReward
from expTypes import RandomDartsExp, SequentialDartsExp, BilliardsExp, BaseballExp, SoccerExp
from pathlib import Path
from gc import collect

# ==== Helper to build infoForEstimators for main() ====
def build_info_for_estimators(args, run_folder_prefix):
    """Build and return the estimators_info dict based on args and domain.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    run_folder_prefix : str
        Run-folder prefix (returned from ensure_output_folders). Used to store rerun start times.

    Returns
    -------
    (estimators_info, list_of_subset_of_estimators)
        estimators_info : dict
            Dictionary ready to pass to createEstimators().
        list_of_subset_of_estimators : list
            Subset of estimators to rerun when args.rerun.
    """

    # Create a list to track the estimators to use
    estimators_list = []

    # If JEEDS is specified, add it to the list of estimators
    if args.jeeds:
        estimators_list = ["JT-QRE"]

    # If the rerun option is specified, set list_of_subset_of_estimators to include BM and log the rerun start time
    # TODO: figure out what list_of_subset_of_estimators is used for
    if args.rerun:
        list_of_subset_of_estimators = ["BM"]
        with open(f"{run_folder_prefix}rerunStartTime{args.seedNum}.txt", 'w') as outfile:
            outfile.write(str(datetime.datetime.now()))
    else:
        list_of_subset_of_estimators = []


    # TODO: This is a hyperparameter used in the AXE estimator. Add notes here explaining it.
    betas = [0.50,0.75,0.85,0.90,0.95,0.99]

    lower_bounds = None
    upper_bounds = None

    # Domain-specific number of hypotheses and reasonable execution skill ranges
    if args.domain == "1d":
        num_execution_skill_hypotheses = [17]
        num_rationality_hypotheses = [33]
        min_execution_skill_for_domain = 0.25
        max_execution_skill_for_domain = 15.0

    elif args.domain in ("2d", "sequentialDarts"):
        num_execution_skill_hypotheses = [33]
        num_rationality_hypotheses = [33]
        min_execution_skill_for_domain = 2.5
        max_execution_skill_for_domain = 150.5

        if args.particles:
            lower_bounds = [2.5,2.5,-0.75,-3]
            upper_bounds = [150.5,150.5,0.75,1.5]

    elif args.domain == "2d-multi":
        num_execution_skill_hypotheses = [[33,33]]
        num_rationality_hypotheses = [33]
        numHypsRhos = 33
        min_execution_skill_for_domain = [2.5,2.5]
        max_execution_skill_for_domain = [150.5,150.5]
        if args.particles:
            lower_bounds = [2.5,2.5,-0.75,-3]
            upper_bounds = [150.5,150.5,0.75,1.5]
    elif args.domain == "billiards":
        num_execution_skill_hypotheses = [2]
        num_rationality_hypotheses = [2]
        min_execution_skill_for_domain = 0.010
        max_execution_skill_for_domain = 0.9
        estimators_list = ["OR","BM","JT-QRE"]
    elif args.domain == "baseball":
        num_execution_skill_hypotheses = [66]
        num_rationality_hypotheses = [66]
        min_execution_skill_for_domain = 0.17
        max_execution_skill_for_domain = 2.81
        minLambda = [1.3, 1.7]
        givenPrior = [[8,0.4,1.0]]
        otherArgs = {"minLambda": minLambda,"givenPrior": givenPrior}
    elif args.domain == "baseball-multi":
        num_execution_skill_hypotheses = [[66,66]]
        num_rationality_hypotheses = [66]
        min_execution_skill_for_domain = [0.17,0.17]
        max_execution_skill_for_domain = [2.81,2.81]
        minLambda = [1.3, 1.7]
        givenPrior = [[8,0.4,1.0]]
        otherArgs = {"minLambda": minLambda,"givenPrior": givenPrior}
        numHypsRhos = 33
        if args.particles:
            lower_bounds = [0.17,0.17,-0.75,-3]
            upper_bounds = [2.81,2.81,0.75,1.5]
    elif args.domain == "soccer":
        num_execution_skill_hypotheses = [2]
        num_rationality_hypotheses = [2]
        min_execution_skill_for_domain = 0.010
        max_execution_skill_for_domain = 0.9
        estimators_list = ["OR","BM","JT-QRE"]
    else:
        # Default fallback
        num_execution_skill_hypotheses = [17]
        num_rationality_hypotheses = [33]
        min_execution_skill_for_domain = 0.25
        max_execution_skill_for_domain = 15.0

    # Particle filter parameters (domain-independent defaults)
    if args.particles:
        if lower_bounds is None or upper_bounds is None:
            if args.domain in ["1d","billiards","soccer"]:
                lower_bounds = [min_execution_skill_for_domain, -3]
                upper_bounds = [max_execution_skill_for_domain, 1.5]
            elif args.domain in ["2d","sequentialDarts"]:
                lower_bounds = [min_execution_skill_for_domain, min_execution_skill_for_domain, -0.75, -3]
                upper_bounds = [max_execution_skill_for_domain, max_execution_skill_for_domain, 0.75, 1.5]
            elif args.domain in ["2d-multi","baseball-multi"]:
                lower_bounds = [min_execution_skill_for_domain[0], min_execution_skill_for_domain[1], -0.75, -3]
                upper_bounds = [max_execution_skill_for_domain[0], max_execution_skill_for_domain[1], 0.75, 1.5]
            elif args.domain == "baseball":
                lower_bounds = [min_execution_skill_for_domain, min_execution_skill_for_domain, -0.75, -3]
                upper_bounds = [max_execution_skill_for_domain, max_execution_skill_for_domain, 0.75, 1.5]
            else:
                lower_bounds = [min_execution_skill_for_domain, -3]
                upper_bounds = [max_execution_skill_for_domain, 1.5]
        estimators_list.append("JT-QRE-Multi-Particles")
        ranges = {"start": lower_bounds, "end": upper_bounds}
        num_particles_list = [int(ii) for ii in args.numParticles] if args.numParticles else [2000]
        noises = [int(ii) for ii in args.noise] if args.noise else [200]
        percents = [float(ii) for ii in args.resample] if args.resample else [0.90]

    # Assemble estimators_info dict
    estimators_info = {"estimators_list": estimators_list,
                         "num_execution_skill_hypotheses": num_execution_skill_hypotheses,
                         "num_rationality_hypotheses": num_rationality_hypotheses,
                         "min_execution_skill_for_domain": min_execution_skill_for_domain,
                         "max_execution_skill_for_domain": max_execution_skill_for_domain,
                         "betas": betas}
    if args.domain in ["baseball","baseball-multi"]:
        estimators_info["otherArgs"] = otherArgs
    if args.domain in ["2d-multi","baseball-multi"]:
        estimators_info["numHypsRhos"] = numHypsRhos
    if args.particles:
        estimators_info.update(
            {
                "diffNs": num_particles_list,
                "noises": noises,
                "percents": percents,
                "resampleNEFF": args.resampleNEFF,
                "ranges": ranges,
                "resamplingMethod": args.resamplingMethod,
            }
        )

    return estimators_info, list_of_subset_of_estimators

# ==== CLI argument parser helper ====
def parse_cli_args():

    parser = argparse.ArgumentParser(
        description='Compute the execution skill for agent given specific number of shots'
    )

    parser.add_argument("-resultsFolder", dest="resultsFolder", help="Enter the name of the folder to store the experiments in", type=str, default="Results")
    parser.add_argument("-domain", dest="domain", help="Specify which domain to use (1d/2d)", type=str, default="2d-multi")
    parser.add_argument("-delta", dest="delta", help="Delta = resolution to use when doing the convolution", type=float, default=1e-2)
    parser.add_argument("-mode", dest="mode", help="Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type=str, default="rand_pos")

    parser.add_argument("-seed", dest="seedNum", help="Enter a seed number", type=int, default=-1)
    parser.add_argument("-folderSeedNums", dest="folderSeedNums", help="Enter the name of the folder containing the desired seedNums file to rerun/read from", type=str, default="test")
    parser.add_argument("-rerun", dest="rerun", help="Flag to rerun exps.", action='store_true')
    parser.add_argument("-noWrap", dest="noWrap", help="Flag to disable wrapping action space in 1D domain.", action='store_true')

    parser.add_argument("-someAgents", dest="someAgents", help="Flag to run exps with only a subset of the agents.", action='store_true')

    parser.add_argument("-saveStates", dest="saveStates", help="To enable the saving of the different set of states used within the experiments.                                                                     Number will be included on file name (needed in case of multiple terminals)", type=int, default=0)
    parser.add_argument("-allProbs", dest="allProbs", help="Flag to enable saving allProbs - probabilities per state - of the BoundedAgent. Info is needed for verifyBoundedAgenPlots.", action='store_true')
    parser.add_argument("-plotState", dest="plotState", help="Flag to enable plotting of states with agent's info (actions & rewards)", action='store_true')

    parser.add_argument("-iters", dest="iters", help="Enter the number of iterations to perform", type=int, default=10)
    parser.add_argument("-numObservations", dest="numObservations", help="Enter the number of observations to use for each experiment.", type=int, default=100)

    parser.add_argument("-xSkillsGiven", dest="xSkillsGiven", help="Flag to enable the use of given params (not rand ones).", action='store_true')
    parser.add_argument("-pSkillsGiven", dest="pSkillsGiven", help="Flag to enable the use of given params (not rand ones).", action='store_true')
    parser.add_argument("-numXskillsPerExp", dest="numXskillsPerExp", help="Enter the number of xskills to use per experiment.", type=int, default=3)
    parser.add_argument("-numPskillsPerExp", dest="numPskillsPerExp", help="Enter the number of pskills to use per experiment.", type=int, default=3)
    parser.add_argument("-numRhosPerExp", dest="numRhosPerExp", help="Enter the number of rhos to use per experiment.", type=int, default=3)

    # FOR DARTS MULTI DOMAIN
    parser.add_argument('-agent', dest='agent', nargs='+', help='Specify the xskill param (per dimension / assuming 2 dimensions) to use for the agent', default=[])

    # FOR BASEBALL DOMAIN
    parser.add_argument("-startYear", dest="startYear", help="Desired start year for the data.", type=str, default="2021")
    parser.add_argument("-endYear", dest="endYear", help="Desired end year for the data.", type=str, default="2021")
    parser.add_argument("-startMonth", dest="startMonth", help="Desired start month for the data.", type=str, default="01")
    parser.add_argument("-endMonth", dest="endMonth", help="Desired end month for the data.", type=str, default="12")
    parser.add_argument("-startDay", dest="startDay", help="Desired start day for the data.", type=str, default="01")
    parser.add_argument("-endDay", dest="endDay", help="Desired end day for the data.", type=str, default="31")
    parser.add_argument('-ids', dest='ids', nargs='+', help='List of pitcher IDs to use', default=[])
    parser.add_argument('-types', dest='types', nargs='+', help='List of pitch types to use', default=[])
    parser.add_argument('-every', dest='every', help='Create checkpoints and reset info every X number of observations.', type=int, default=20)
    parser.add_argument("-maxRows", dest="maxRows", help="Max number of most recent pitches to select from data.",  type=int, default=1000)
    parser.add_argument("-reload", dest="reload", help="Flag to reload row info from prev exps.", action='store_true')
    parser.add_argument("-dataBy", dest="dataBy", help="Flag to specify how to get/filter the data by (recent,chunks,pitchNum).", type=str, default="recent")

    # FOR PFE
    parser.add_argument('-numParticles', dest='numParticles', nargs='+', help='List containing the different number of particles to test', default=[2000])
    parser.add_argument("-particles", dest="particles", help="Flag to enable the use of estimators with particle filter.", action='store_true')
    parser.add_argument("-resampleNEFF", dest="resampleNEFF", help="Flag to enable resampling based on the NEFF threshold", action='store_true')
    parser.add_argument("-resample", dest="resample", help="Specify the percent to use for the resampling", nargs='+', default=[])
    parser.add_argument('-noise', dest='noise', nargs='+', help='List of noises to use (W/noise)', default=[])
    parser.add_argument('-resamplingMethod', dest='resamplingMethod', help='Specify the resampling method to use for the PF (Default = numpy choice).', type=str, default="numpy")

    # For Dynamic Agents
    parser.add_argument("-dynamic", dest="dynamic", help="Flag to enable agents to have dynamic xskill", action='store_true')

    # FOR ESTIMATORS
    parser.add_argument("-jeeds", dest="jeeds", help="Flag to enable the use of jeeds estimator only.", action='store_true')
    parser.add_argument("-pfe", dest="pfe", help="Flag to enable the use of given pfe estimator only.", action='store_true')
    parser.add_argument("-pfeNeff", dest="pfeNeff", help="Flag to enable the use of pfe estimators (with neff sampling) only.", action='store_true')

    return parser.parse_args()

_PERSISTENT_DOMAINS = {"baseball", "baseball-multi", "soccer"}


def run_single_experiment(exp, tag, counter, domain_name, result_queue):
    """
    Run a single experiment (possibly in a child process) and optionally send results back.

    Parameters
    ----------
    exp : object
        Experiment object exposing `run(tag, counter)`, `getResults()`, and `getValid()`/`getStatus()` (used elsewhere).
    tag : str
        Identifier string used to tag files and logs for this run.
    counter : int
        Sequential counter to disambiguate multiple runs within the same iteration.
    domain_name : str
        Name of the current domain. For domains that persist results internally
        (see `_PERSISTENT_DOMAINS`), no results are pushed to the queue.
    result_queue : Queue
        A `multiprocess.Queue` provided by the parent process to receive results.
        It may be `None` for persistent domains where nothing is sent back.

    Behavior
    --------
    - Always calls `exp.run(tag, counter)`.
    - If the domain does *not* persist its results internally, obtains the results via
      `exp.getResults()` and pushes them to `result_queue` for the parent to consume.
    - If an exception occurs, attempts to notify the parent by placing a small error
      dictionary on the queue (if available) and then re-raises the exception so the
      worker process exits with a non-zero status.
    """
    try:
        # Execute the experiment
        exp.run(tag, counter)

        # For most domains, results are returned to the parent via the queue.
        if domain_name not in _PERSISTENT_DOMAINS:
            if result_queue is not None:
                result_queue.put(exp.getResults())
            # If result_queue is None, we silently skip returning results; the parent should
            # not expect them in these code paths.

    except Exception as exc:
        # Best-effort error propagation back to the parent process
        if result_queue is not None:
            try:
                result_queue.put({"__error__": f"work() failed: {exc!r}"})
            except Exception:
                pass  # Avoid masking the original exception
        # Re-raise so the child process terminates with an error
        raise


def _compute_labels_and_paths(args, agent, tag, counter, env):
    """Return (label, results_file, status_file) based on domain and agent.

    Keeps the original file-naming and status-file conventions intact.
    """
    if args.domain == "billiards":
        label = f"Agent: {agent}"
        save_at = f"{tag}-{counter}-Agent{agent}.results"
    elif args.domain in ["baseball", "baseball-multi"]:
        label = f"Agent -> pitcherID: {agent[0]} | pitchType: {agent[1]}"
        save_at = f"{tag}.results"
    elif args.domain == "soccer":
        label = f"Agent -> playerID: {agent}"
        save_at = f"{tag}.results"
    else:
        label = f"Agent: {agent.name}"
        save_at = f"{tag}-{counter}-Agent{agent.getName()}.results"

    results_file = f"Experiments{os.sep}{args.resultsFolder}{os.sep}results{os.sep}{save_at}"

    if args.domain in ["baseball", "baseball-multi", "soccer"]:
        status_file = f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}{tag}-DONE"
    else:
        status_file = (
            f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}"
            f"{tag}-{counter}-Agent{agent.getName()}-DONE"
        )

    return label, results_file, status_file


def _make_experiment(env, args, agent, xskill, estimators_obj, subset_estimators,
                    results_file, index_or, seed_num, rng, temp_rerun):
    """Construct the correct experiment object for the current domain.

    Mirrors the original if/elif ladder but centralizes it for readability.
    """
    if env.domainName in ["1d", "2d", "2d-multi"]:
        return RandomDartsExp(env.numObservations, args.mode, env, agent, xskill,
                              estimators_obj, subset_estimators, args.resultsFolder,
                              results_file, index_or, args.allProbs, seed_num, rng, temp_rerun)
    elif env.domainName == "sequentialDarts":
        return SequentialDartsExp(env.numObservations, args.mode, env, agent, xskill,
                                  estimators_obj, subset_estimators, args.resultsFolder,
                                  results_file, index_or, args.allProbs, seed_num, rng, temp_rerun)
    elif env.domainName == "billiards":
        return BilliardsExp(env.numObservations, env, agent, xskill,
                            estimators_obj, subset_estimators, args.resultsFolder,
                            results_file, index_or, seed_num, rng, temp_rerun)
    elif env.domainName in ["baseball", "baseball-multi"]:
        return BaseballExp(args, env, agent, estimators_obj, subset_estimators,
                           args.resultsFolder, results_file, index_or, seed_num, rng)
    elif env.domainName == "soccer":
        return SoccerExp(args, env, agent, estimators_obj, subset_estimators,
                         args.resultsFolder, results_file, index_or, seed_num, rng)
    else:
        raise ValueError(f"Unknown domainName: {env.domainName}")



def _run_exp_and_collect_results(env, exp, tag, counter):
    """Execute the experiment and return a results dict (may be empty).

    - For persistent-result domains (baseball / baseball-multi / soccer), this runs inline
      and returns an empty dict, since those experiments persist to disk internally.
    - For other domains, it runs the experiment in a child process via `run_single_experiment`
      and returns the results placed on the queue by the worker.
    """
    if env.domainName in ["baseball", "baseball-multi", "soccer"]:
        exp.run(tag, counter)
        return {}

    result_queue = Queue()
    process = Process(target=run_single_experiment,
                      args=(exp, tag, counter, env.domainName, result_queue))
    process.start()
    results = result_queue.get()
    process.join()
    return results

# ==== OnlineExperiment helpers (factored out for clarity) ====

def _should_skip_completed(env, args, status_file, label):
    """Return True if this experiment can be skipped as already completed.

    Skips only for persistent-result domains (baseball/baseball-multi) when
    a DONE marker exists and args.rerun is False. Also frees large cached data
    on the env when possible to keep memory stable across many agents.
    """
    if env.domainName in ["baseball", "baseball-multi"] and Path(f"{status_file}.txt").is_file() and not args.rerun:
        print(f"Experiment for {label} was already performed and it finished successfully.")
        try:
            del env.spaces.allData
        except Exception:
            pass
        return True
    return False


def _validate_rerun_flag(results_file, requested_rerun):
    """Only keep rerun=True if a prior results file exists and is readable."""
    if not requested_rerun:
        return False
    try:
        with open(results_file, "rb") as handle:
            _ = pickle.load(handle)
        return True
    except Exception:
        return False


def _run_and_time(env, exp, tag, counter):
    """Run the experiment and return (results_dict, elapsed_seconds)."""
    start = time.time()
    results = _run_exp_and_collect_results(env, exp, tag, counter)
    return results, (time.time() - start)


def _persist_results(env, results_file, results, exp_total_time):
    """Write timing/metadata and merge results (when applicable) to disk."""
    import datetime as _dt

    if env.domainName not in ["baseball", "baseball-multi", "soccer"]:
        # Non-persistent domains: merge returned results with timing
        with open(results_file, "rb") as handle:
            results_loaded = pickle.load(handle)
        results["expTotalTime"] = exp_total_time
        results["lastEdited"] = str(_dt.datetime.now())
        results_loaded.update(results)
        with open(results_file, "wb") as outfile:
            pickle.dump(results_loaded, outfile)
        del results_loaded
    else:
        # Persistent domains: experiment already wrote results; append timing only
        with open(results_file, "rb") as handle:
            persisted = pickle.load(handle)
        persisted["expTotalTime"] = exp_total_time
        with open(results_file, "wb") as outfile:
            pickle.dump(persisted, outfile)
        del persisted


def _mark_success(status_file, args, temp_rerun):
    """Create DONE markers; add extra marker in rerun mode for book-keeping."""
    # Primary DONE marker
    Path(f"{status_file}.txt").write_text("")

    if temp_rerun is True:
        current_files = os.listdir(
            f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}"
        )
        extra_status = f"{status_file}-RERUN-{len(current_files)}.txt"
        Path(extra_status).write_text("")


def _cleanup_after_agent(exp):
    """Free per-agent resources and encourage GC; ignore any GC errors."""
    del exp
    try:
        collect()
    except Exception:
        pass

# ==== Output folder setup helper ====
def ensure_output_folders(args) -> str:
    """Create the experiment folder structure and normalize args.resultsFolder.

    Returns
    -------
    str
        The computed run-folder path `results_folder_path`, e.g.,
        "Experiments/<domain>/<resultsFolder>/".
    """
    results_folder_path = f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}"

    folders = [
        "Experiments",
        f"Experiments{os.sep}{args.domain}",
        results_folder_path,
        f"{results_folder_path}results{os.sep}",
        "Data",
        "Spaces",
        f"Spaces{os.sep}ExpectedRewards",
        f"{results_folder_path}status{os.sep}",
        f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}times{os.sep}",
        f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}times{os.sep}experiments",
    ]

    if args.domain == "sequentialDarts":
        folders += [f"Spaces{os.sep}ValueFunctions"]

    if args.domain == "billiards":
        f = f"Data{os.sep}{args.domain.capitalize()}{os.sep}"
        folders += [f"{f}ProcessedShots", f"Spaces{os.sep}SuccessRates"]

    if args.domain in ["baseball", "baseball-multi"]:
        f = f"Data{os.sep}{args.domain.capitalize()}{os.sep}"
        f2 = f"{results_folder_path}plots{os.sep}"
        folders += [
            f"{results_folder_path}results{os.sep}",
            f"{results_folder_path}info{os.sep}",
            f,
            f2,
            f"{f2}StrikeZoneBoards{os.sep}",
            f"{f2}StrikeZoneBoards{os.sep}PickleFiles{os.sep}",
            f"{f}Models{os.sep}",
            f"{f}StatcastData{os.sep}",
            f"{results_folder_path}results{os.sep}backup{os.sep}",
            f"{results_folder_path}info{os.sep}backup{os.sep}",
        ]

    if args.domain == "soccer":
        folders += [
            f"Data{os.sep}Soccer{os.sep}",
            f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}",
            f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}Dataset{os.sep}",
        ]

    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except FileExistsError:
                continue

    # Normalize resultsFolder to include the domain prefix
    args.resultsFolder = args.domain + os.sep + args.resultsFolder

    return results_folder_path

# ==== End helpers ====


# ==== Experiment context preparation helper ====
def prepare_experiment_context(args):
    """Initialize seeds, environment, and agents; compute number of experiments.

    Returns
    -------
    (env, agents, seeds, num_experiments)
        env : Environment
            Newly created environment configured by `args`.
        agents : list | None
            Agent list for domains that predefine agents (baseball/baseball-multi/soccer);
            otherwise None and agents are created later per x-skill.
        seeds : np.ndarray
            Per-iteration RNG seeds derived from the main seed.
        num_experiments : int
            Number of iterations to run.
    """
    # Ensure a main seed; -1 means "randomize now"
    if args.seedNum == -1:
        args.seedNum = np.random.randint(0, 100000, 1)[0]

    # Seed numpy's global RNG for any legacy callers
    np.random.seed(args.seedNum)
    print(f"Seed set to: {args.seedNum}")

    # Generate one seed per iteration for downstream use
    seeds = np.random.randint(0, 1_000_000, args.iters)

    print("\nSetting environment...")
    env = setupEnv.Environment(args)

    agents = None
    if args.domain in ["baseball", "baseball-multi", "soccer"]:
        agents = env.agentGenerator.getAgents()

    # Number of experiments is iterations for most domains, else number of agents
    if args.domain not in ["baseball", "baseball-multi", "soccer"]:
        num_experiments = args.iters
    else:
        num_experiments = len(agents)

    return env, agents, seeds, num_experiments


# # ==== Cluster experiment defaults helper ====
# def apply_cluster_defaults(args) -> None:
#     """Apply the hard-coded defaults used for cluster experiments.
#
#     This preserves the original behavior from `main()` by setting a series of
#     arguments related to iteration counts, estimator toggles, and result
#     folder naming. No return value; modifies `args` in place.
#     """
#     # Iterations and observations
#     args.iters = 500
#     args.numObservations = 100
#
#     # Use given (non-random) x/p skills and counts
#     args.xSkillsGiven = True
#     args.pSkillsGiven = True
#     args.numXskillsPerExp = 3
#     args.numPskillsPerExp = 3
#     args.numRhosPerExp = 3
#
#     # Particle filter defaults (unless JEEDS flag is set)
#     args.particles = True
#     if args.jeeds:
#         args.particles = False
#
#     # NEFF-based resampling defaults
#     args.pfeNeff = True
#     args.resampleNEFF = True
#
#     # Name the results folder consistently with previous convention
#     args.resultsFolder = (
#         f"Experiments-Round2-NumObservations{args.numObservations}-"
#         f"NumParticles{args.numParticles[0]}"
#     )
#
#     # NOTE: There was an optional testing toggle for dynamic agents left commented
#     # in main(). We preserve that comment in-place and do not enable it here.


# ==== Legacy memory-management helper (no-op) ====
def cleanup_memory_placeholders(args) -> None:
    """Legacy cleanup step extracted from main().

    Historically, main() attempted to `del` several locals like `namesEstimators`,
    `startX_Estimator`, and `stopX_Estimator` to hint the GC. After refactors,
    those values live inside helper scopes and aren't present here; deleting them
    would raise NameError. We keep a named helper for readability and future use,
    but it intentionally does nothing.
    """
    # Intentionally left as a no-op. If you later reintroduce large local
    # temporaries into main(), you can explicitly `del` them here.
    return None


# ==== Helper to locate ObservedReward estimator index ====
def find_observed_reward_index(estimators) -> int | None:
    """Return the index of the ObservedReward estimator if present, else None."""
    try:
        temp = estimators.getCopyOfEstimators()
    except Exception:
        return None
    for idx, est in enumerate(temp):
        if isinstance(est, ObservedReward):
            return idx
    return None

# ==== Per-iteration environment preparation helper ====
def prepare_iteration_spaces(env, args, estimators, rng, iteration_index: int) -> None:
    """Reset env and (re)build spaces for a new iteration.

    This groups the repeated steps from main():
      - regenerate agent generator params if x/p skills are randomized
      - set new states for darts-like domains
      - reset env bookkeeping
      - set agent spaces
      - set estimator spaces with domain-specific rules
    """
    # For darts-like domains, (re)randomize agent generator params when not given
    if args.domain in ["1d", "2d", "2d-multi", "sequentialDarts"]:
        if not env.xSkillsGiven:
            env.agentGenerator.setXskillParams()
        if not env.pSkillsGiven:
            env.agentGenerator.setPskillParams()

    # For 1d/2d/2d-multi, generate a fresh set of states per iteration
    if args.domain in ["1d", "2d", "2d-multi"]:
        env.setStates(rng)

    # Reset env structures that depend on states/rows
    env.resetEnv()

    # Update spaces to account for new states and agent params
    env.setSpacesAgents(rng)

    # Estimator spaces depend on domain
    if args.domain in ["1d", "2d"]:
        env.setSpacesEstimators(rng, args, estimators.allXskills)
    elif args.domain in ["2d-multi", "baseball-multi"]:
        env.setSpacesEstimators(rng, args, [estimators.allXskills, estimators.rhos])
    else:
        # Other domains set estimator spaces once at the beginning
        if iteration_index == 0:
            env.setSpacesEstimators(rng, args, estimators.allXskills)


# ==== Iteration tag builder & rerun resolver ====
def build_iteration_tag(args, each, justXs: str, seedNum: int, iteration_index: int, rerun_search_dir: str):
    """Construct the experiment tag for an iteration and determine rerun mode.

    Preserves original behavior, including:
      - Searching prior results to recover the original time-based tag when --rerun
      - Fallback to fresh time-based tag if not found
      - Appending JEEDS/PFE/PFE_NEFF and resampling method suffixes as before

    Returns
    -------
    (tag: str, expRerun: bool)
    """
    import os
    # Default values
    tag = None
    expRerun = False

    if args.rerun:
        currentFiles = os.listdir(rerun_search_dir)

        # Try to reproduce previous x-skill formatting; handle scalars vs sequences
        try:
            xStr = str(round(each, 3)).replace('.', '-')  # scalar case
        except Exception:
            # Fallback: use the already-built justXs string
            xStr = justXs.replace('.', '-').strip('-')

        # Locate a prior result filename containing both xStr and seedNum
        for eachRF in currentFiles:
            if xStr in eachRF and str(seedNum) in eachRF:
                tag = eachRF.split(f"{seedNum}")[0] + f"{seedNum}"
                expRerun = True
                print(f"Found tag: {tag}")
                break

        if tag is None:
            print("On rerun mode but tag not found. Proceeding to perform full experiment.")
            base = xStr
            tag = f"OnlineExp_{base}_{time.strftime('%y_%m_%d_%M_%S')}_{iteration_index}_{seedNum}"
    else:
        base = justXs.replace('.', '-')
        tag = f"OnlineExp_{base}_{time.strftime('%y_%m_%d_%M_%S')}_{iteration_index}_{seedNum}"
        if args.particles:
            tag += f"_{args.resamplingMethod}"

    # Append method flags
    if args.jeeds:
        tag += "_JEEDS"
    if args.pfe:
        tag += "_PFE"
    if args.pfeNeff:
        tag += "_PFE_NEFF"

    return tag, expRerun

# ==== Helper to format x-skill display strings for logging/tags ====

def format_xskill_strings(each, args):
    """Return (aStr, justXs) for the current x-skill selection.

    This preserves existing formatting, including multi-dim and dynamic cases
    in the 2d-multi domain, and the simple scalar formatting otherwise.
    """
    aStr = ""
    justXs = ""

    if args.domain == "2d-multi":
        for ti in range(len(each)):
            if ti == 1:
                # Rho component printed on a new labeled line
                aStr += "\nRHO:"
                aStr += f" {each[ti]} | "
                justXs += f"{each[ti]}-"
            else:
                if args.dynamic:
                    # Dynamic agents carry (start, end) tuples per dimension
                    aStr += f" Start: {each[ti][0]} | End: {each[ti][1]}"
                    justXs += f"{each[ti][0]}-{each[ti][1]}"
                else:
                    # Static: print all entries for this dimension
                    for t in each[ti]:
                        aStr += f" {t} | "
                        justXs += f"{t}-"
    else:
        aStr += f" {each} | "
        justXs += f"{each}-"

    return aStr, justXs

# ==== Helper to update spaces for 2d-multi domain ====
def update_space_for_2d_multi(env, rng, each, dynamic: bool) -> None:
    """Update env.spaces for a single 2d-multi agent configuration.

    Mirrors the original behavior:
      - If dynamic, update twice (for each dimension's start/end tuple) with rho
      - Otherwise, update once using `[xskills, [rho]]`
    """
    if dynamic:
        env.spaces.updateSpace(rng, [[each[0][0]], [each[-1]]], env.states)
        env.spaces.updateSpace(rng, [[each[0][1]], [each[-1]]], env.states)
    else:
        env.spaces.updateSpace(rng, [each[:-1], [each[-1]]], env.states)


# ==== Per-iteration timing & persistence helper ====
def record_iteration_time(args, timesPerExps, overallStart, expStart, iteration_index: int) -> None:
    """Record timing for an iteration, print summary, and persist rolling stats.

    Side effects:
      - Appends elapsed time to `timesPerExps`
      - Prints per-iteration and running-average timings
      - Writes a pickle with {"timesPerExps", "avgTimePerExps"} to the times/experiments folder
    """
    expStop = time.time()
    timesPerExps.append(expStop - expStart)

    print(f"Finished Online Iteration: {iteration_index}")
    print(
        f"Total Time elapsed: {expStop - overallStart} seconds.  "
        f"Average of {(expStop - overallStart) / float(iteration_index + 1)} per iteration.\n"
    )
    print("-" * 70)
    print()

    # Persist timing results on every iteration
    timesResults = {
        "timesPerExps": timesPerExps,
        "avgTimePerExps": sum(timesPerExps) / len(timesPerExps),
    }

    with open(
        f"Experiments{os.sep}{args.resultsFolder}{os.sep}times{os.sep}experiments{os.sep}times-{args.seedNum}",
        "wb",
    ) as outfile:
        pickle.dump(timesResults, outfile)



# ==== Persistent-domain iteration runner (baseball / baseball-multi / soccer) ====
def run_persistent_domain_iteration(args, env, agents, estimators, subsetEstimators,
                                   counter, seedNum, rng, indexOR, iteration_index: int) -> int:
    """Build tag for persistent domains and run `onlineExperiment` for a single agent.

    Behavior mirrors the inline code previously in `main()`:
      - Picks the agent for this iteration from `agents`
      - Constructs the tag depending on domain and method flags
      - Calls `onlineExperiment` with xskill=None and a single-agent list
      - Returns the updated counter
    """
    # Agent selection per iteration
    agent = agents[iteration_index]

    # Base tag depends on domain
    if args.domain in ["baseball", "baseball-multi"]:
        tag = f"OnlineExp_{agent[0]}_{agent[1]}"
    else:  # soccer
        tag = f"OnlineExp_{agent}"

    # Append method flags (JEEDS / PFE / PFE_NEFF)
    if args.jeeds:
        tag += f"_JEEDS_Chunk_{args.seedNum}"
    if args.pfe:
        tag += f"_PFE_Chunk_{args.seedNum}"
    if args.pfeNeff:
        tag += f"_PFE_NEFF_Chunk_{args.seedNum}"

    # NOTE: args.seedNum indicates the chunk number for these domains
    return onlineExperiment(
        args,
        None,
        [agent],
        env,
        estimators,
        subsetEstimators,
        tag,
        counter,
        seedNum,
        rng,
        indexOR,
    )

# ==== Helper to assemble per-iteration x-skill configurations ====
def build_all_xskill_info(env, args):
    """Return the iterable of x-skill configurations for this iteration.

    - For 2d-multi: returns cartesian product of xskills (or dynamicXskills) and rhos
    - For other domains: returns the list of xSkills provided by the agentGenerator
    """
    if args.domain == "2d-multi":
        # Local import to avoid requiring global imports
        from itertools import product
        if args.dynamic:
            return list(product(env.agentGenerator.dynamicXskills, env.agentGenerator.rhos))
        return list(product(env.agentGenerator.xSkills, env.agentGenerator.rhos))
    else:
        return env.agentGenerator.xSkills


def onlineExperiment(
    args,
    xskill,
    agents,
    env,
    estimatorsObj,
    subsetEstimators,
    tag,
    counter,
    seedNum,
    rng,
    indexOR,
    rerun: bool = False,
) -> int:
    """
    Run one pass of experiments for the provided agents.

    This orchestrator keeps behavior identical while delegating to small helpers for:
      - skip logic for already-completed runs
      - rerun validation
      - running & timing
      - result persistence
      - status marking
      - cleanup
    """
    print("\nPerforming experiments...\n")

    for agent in agents:
        # New RNG per agent (preserves original behavior)
        agent_rng = np.random.default_rng(seedNum)

        label, results_file, status_file = _compute_labels_and_paths(
            args, agent, tag, counter, env
        )

        # Early exit for already-completed persistent domains (unless rerun)
        if _should_skip_completed(env, args, status_file, label):
            counter += 1
            continue

        # Rerun only if a prior results file exists and is readable
        temp_rerun = _validate_rerun_flag(results_file, rerun)

        # Construct experiment
        exp = _make_experiment(
            env,
            args,
            agent,
            xskill,
            estimatorsObj,
            subsetEstimators,
            results_file,
            indexOR,
            seedNum,
            agent_rng,
            temp_rerun,
        )

        if exp.getValid():
            results, elapsed = _run_and_time(env, exp, tag, counter)
            _persist_results(env, results_file, results, elapsed)
            print(f"Total time for experiment: {elapsed:.4f}")

        # Mark success if experiment reports a good status
        if exp.getStatus():
            _mark_success(status_file, args, temp_rerun)

        counter += 1
        _cleanup_after_agent(exp)

    return counter


def createEstimators(args,infoForEstimators,env):

    print("\nCreating estimators...")

    creationStart = time.time()

    estimators = EstimatorsClass(infoForEstimators, env)

    #
    # code.interact("...", local=dict(globals(), **locals()))

    creationStop = time.time()

    print("Done creating the estimators.\n")

    # store the time taken to create estimators to txt file
    with open("Experiments" + os.sep + args.resultsFolder  + os.sep + "times" + os.sep + f"timeToCreateEstimators{args.seedNum}.txt", "w") as file:
        file.write("\n\nTime to create all the estimators: "+str(creationStop - creationStart))

    return estimators


# ==== Estimator creation + OR-index discovery helper ====
def create_estimators_and_index(args, info_for_estimators, env):
    """Create estimators and return (estimators, indexOR) honoring useEstimators flag."""
    estimators = createEstimators(args, info_for_estimators, env)
    _index_obs_reward = find_observed_reward_index(estimators)
    return estimators, _index_obs_reward


# ==== Whole-experiment runner (loops over iterations and x-skill configs) ====
def run_all_experiments(args, env, agents, estimators, subsetEstimators,
                        seeds, numExperiments, indexOR):
    """Run all iterations across persistent/non-persistent domains with timing."""
    counter = 0
    timesPerExps = []
    overallStart = time.time()

    for i in range(numExperiments):
        print(f"{'-'*30} START ITERATION {i} {'-'*30}")
        expStart = time.time()

        seedNum = int(seeds[i])
        rng = np.random.default_rng(seeds[i])

        # Prepare env state/spaces for this iteration
        prepare_iteration_spaces(env, args, estimators, rng, i)

        # Perform experiment
        if args.domain in ["baseball", "baseball-multi", "soccer"]:
            counter = run_persistent_domain_iteration(
                args, env, agents, estimators, subsetEstimators,
                counter, seedNum, rng, indexOR, i
            )
        else:
            allInfo = build_all_xskill_info(env, args)
            for each in allInfo:
                print(f"{'-'*70}\nITERATION {i} - \nEXECUTION SKILL LEVEL: ", end="")
                aStr, justXs = format_xskill_strings(each, args)
                print(aStr)
                print(f"{'-'*70}")

                if args.domain == "2d-multi":
                    update_space_for_2d_multi(env, rng, each, args.dynamic)

                # Agents for this x-skill setting
                agents_local = env.agentGenerator.getAgents(rng, env, each)

                # Build tag and determine rerun status
                search_dir = f"Experiments{os.sep}{args.resultsFolder}{os.sep}results{os.sep}"
                tag, expRerun = build_iteration_tag(args, each, justXs, seedNum, i, search_dir)

                counter = onlineExperiment(
                    args, each, agents_local, env, estimators, subsetEstimators,
                    tag, counter, seedNum, rng, indexOR, expRerun
                )

        # Record timing and persist rolling stats
        record_iteration_time(args, timesPerExps, overallStart, expStart, i)

    return counter


# ==== Seed persistence helper ====
def save_seeds_if_needed(args, seeds) -> None:
    """Persist seeds used for the run unless in rerun mode."""
    if args.rerun:
        return
    results = {"seeds": seeds.tolist()}
    with open(
        f"Experiments{os.sep}{args.resultsFolder}{os.sep}seeds{args.seedNum}",
        "wb",
    ) as outfile:
        pickle.dump(results, outfile)


# @profile
def main():
    # Get arguments from command line
    args = parse_cli_args()

    # Apply cluster experiment defaults
    # apply_cluster_defaults(args) # TODO: Figure out why this existed and then rework it

    # Initial Setup
    _results_folder_path = ensure_output_folders(args)

    # Build info_for_estimators and subset_estimators
    info_for_estimators, subset_estimators = build_info_for_estimators(args, _results_folder_path)

    # Prepare environment, agents, seeds, and iteration count
    env, agents, seeds, num_experiments = prepare_experiment_context(args)

    # Create the estimators
    estimators, index_obs_reward = create_estimators_and_index(args, info_for_estimators, env)

    # Memory management (legacy no-op; kept as a named step for clarity)
    cleanup_memory_placeholders(args)

    run_all_experiments(args, env, agents, estimators, subset_estimators, seeds, num_experiments, index_obs_reward)

    # Store seeds used (if not on rerun mode, as seeds will be provided if that's the case)
    save_seeds_if_needed(args, seeds)

    print('\n****\nDone with all', num_experiments, 'experiments\n****')
    # code.interact("END...", local=dict(globals(), **locals()))



if __name__ == "__main__": 
    main()




