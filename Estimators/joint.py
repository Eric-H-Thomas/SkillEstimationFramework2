import numpy as np
import time
import os, sys
from scipy.stats import norm, skewnorm
import gc

gc.set_debug(gc.DEBUG_SAVEALL)

try:
    from Estimators.utils import *
except:
    from importlib.machinery import SourceFileLoader

    scriptPath = os.path.realpath(__file__)
    mainFolderName = scriptPath.split("Estimators")[0]
    sys.path.insert(1, mainFolderName)
    from Estimators.utils import *
    from Estimators.joint_pfe import *


class JointMethodFlip:
    def __init__(self, xskills, numPskills, domainName):

        self.xskills = xskills
        self.numXskills = len(xskills)
        self.numPskills = numPskills

        self.domainName = domainName

        self.methodType = "JT-FLIP"

        self.names = ['JT-FLIP-MAP-' + str(self.numXskills), 'JT-FLIP-EES-' + str(self.numXskills)]

        self.estimatesXskills = dict()
        self.estimatesPskills = dict()

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesPskills[n] = []

        if domainName == "1d":
            self.pskills = np.logspace(-3, 2, self.numPskills)  # 0.001 to 100
        elif domainName in ["2d", "sequentialDarts", "billiards"]:
            self.pskills = np.logspace(-3, 1.5, self.numPskills)
        elif domainName == "baseball":
            self.pskills = np.logspace(-3, 3.6, self.numPskills)

        self.probs = np.ndarray(shape=(self.numXskills, self.numPskills))

        # initializing the array
        self.probs.fill(1.0 / (self.numXskills * self.numPskills * 1.0))

        self.allProbs = []

        # to append the initial probs - uniform distribution for all
        self.allProbs.append(self.probs.tolist())

    def getEstimatorName(self):
        """Return the metric names used for this estimator (MAP and EES variants)."""
        return self.names

    def reset(self):
        for n in self.names:
            self.estimatesPskills[n] = []
            self.estimatesXskills[n] = []

        # reset probs
        self.probs.fill(1.0 / (self.numXskills * self.numPskills * 1.0))

        self.allProbs = []

        self.allProbs.append(self.probs.tolist())

    def addObservation(self, spaces, action, **otherArgs):

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialize info
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        resultsFolder = otherArgs["resultsFolder"]
        tag = otherArgs["tag"]
        delta = spaces.delta

        if self.domainName == "sequentialDarts":
            # Sequential darts keeps track of the running score to select the correct state-space slice
            currentScore = otherArgs["currentScore"]

        action = np.array(action)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        startTimeEst = time.perf_counter()

        if self.domainName == "sequentialDarts":
            pdfs = computePDF(x=action, means=spaces.allPIsForXskillsPerState[currentScore], covs=spaces.all_covs)

        for xi in range(self.numXskills):

            # Get the corresponding xskill level at the given index
            x = self.xskills[xi]

            ###########################################################################################
            # COMPUTING PDFS
            ###########################################################################################

            if self.domainName == "1d" or self.domainName == "2d":
                space = spaces.convolutionsPerXskill[x][otherArgs["s"]]
                targetAction = space["ts"]

            elif self.domainName == "sequentialDarts":
                space = spaces.spacesPerXskill[x]

            elif self.domainName == "billiards":
                diff = otherArgs["diff"]

            elif self.domainName == "baseball":
                pdfs = computePDF(x=action, means=spaces.possibleTargetsFeet,
                                  covs=[spaces.all_covs[xi]] * len(spaces.possibleTargetsFeet))

            for pi in range(len(self.pskills)):

                # Get the corresponding pskill level at the given index
                p = self.pskills[pi]

                # Update probs - flip planning component like
                if self.domainName == "1d":
                    diff_fn = getattr(spaces.domain, "calculate_wrapped_action_difference", None)
                    if diff_fn is None:
                        diff_fn = getattr(spaces.domain, "calculate_action_difference")
                    action_diff = diff_fn(action, targetAction)
                    self.probs[xi][pi] *= ((p * scipy.stats.norm.pdf(action_diff, loc=0, scale=x)) + (
                                (1 - p) / spaces.sizeActionSpace))

                elif self.domainName == "2d":
                    self.probs[xi][pi] *= ((p * (multivariate_normal.pdf(x=action, mean=targetAction, cov=(x ** 2)) * (
                                spaces.delta ** 2))) + ((1 - p) / spaces.sizeActionSpace))

                elif self.domainName == "sequentialDarts":
                    self.probs[xi][pi] *= (p * pdfs[xi] * (delta ** 2)) + ((1 - p) / spaces.sizeActionSpace)

                elif self.domainName == "billiards":
                    diff = otherArgs["diff"]
                    self.probs[xi][pi] *= (p * scipy.stats.norm.pdf(x=diff, loc=0, scale=x)) + (
                                (1 - p) / spaces.sizeActionSpace)

                elif self.domainName == "baseball":
                    self.probs[xi][pi] *= (p * pdfs[xi] + ((1 - p) / spaces.sizeActionSpace))

        # Normalize
        self.probs /= np.sum(self.probs)
        self.allProbs.append(self.probs.tolist())

        # Get estimate. Uses MAP estimate
        # Get index of maximum prob - returns flat index
        mi = np.argmax(self.probs)

        # "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
        xmi, pmi = np.unravel_index(mi, self.probs.shape)

        self.estimatesXskills[self.names[0]].append(self.xskills[xmi])
        self.estimatesPskills[self.names[0]].append(self.pskills[pmi])

        # Get EES & EPS Estimate
        ees = 0.0
        eps = 0.0

        for xi in range(self.numXskills):

            for pi in range(self.numPskills):
                ees += self.xskills[xi] * self.probs[xi][pi]
                eps += self.pskills[pi] * self.probs[xi][pi]

        self.estimatesXskills[self.names[1]].append(ees)
        self.estimatesPskills[self.names[1]].append(eps)

        endTimeEst = time.perf_counter()
        totalTimeEst = endTimeEst - startTimeEst

        # print "JT-FLIP"
        # print "EES: ", ees, "\t\t MAP: ", self.estimatesXskills[self.names[0]][-1]
        # print "EPS: ", eps, "\t\t MAP: ", self.estimatesPskills[self.names[0]][-1], "\n"
        # code.interact("", local=locals())

        folder = resultsFolder

        # If the plots folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep)

        # If the folder doesn't exist already, create it
        if not os.path.exists(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators"):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators")

        with open(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators" + os.path.sep + "JT-FLIP-Times-" + tag,
                "a") as file:
            file.write(str(totalTimeEst) + "\n")

    def getResults(self):
        results = dict()

        for n in self.names:
            results[n + "-pSkills"] = self.estimatesPskills[n]
            results[n + "-xSkills"] = self.estimatesXskills[n]

        results[f"{self.methodType}-allProbs"] = self.allProbs

        return results


# Note: JEEDS is the same thing as JointMethodQRE
class JointMethodQRE:

    @staticmethod
    def _build_label(given_prior, min_lambda, other_args):
        """Return a suffix label describing which specialized priors were used."""

        if given_prior and min_lambda:
            return (
                f"-GivenPrior-{other_args['givenPrior'][0]}-{other_args['givenPrior'][1]}-{other_args['givenPrior'][2]}"
                f"-MinLambda{other_args['minLambda']}"
            )
        if given_prior:
            return f"-GivenPrior-{other_args['givenPrior'][0]}-{other_args['givenPrior'][1]}-{other_args['givenPrior'][2]}"
        if min_lambda:
            return f"-MinLambda-{other_args['minLambda']}"
        return ""

    def __init__(self, execution_skills, num_rationality_levels, domain_name, given_prior=False, min_lambda=False,
                 other_args=None, *, times_base_dir=None):
        """Initialize a joint estimator over execution skills and rationality levels.

        Parameters
        ----------
        execution_skills : list[float]
                Grid of execution skill hypotheses.
        num_rationality_levels : int
                Number of rationality level hypotheses (lambda values).
        domain_name : str
                Name of the environment so the estimator can configure priors correctly.
        given_prior : bool
                Whether to initialize a skew-normal prior over execution skills.
        min_lambda : bool
                Whether to shift the minimum lambda value when building rationality priors.
        other_args : dict | None
                Additional tuning parameters used when givenPrior/minLambda are enabled.
        times_base_dir : str | None
                If set, timing logs are written under ``<times_base_dir>/times/estimators/``
                instead of the default ``Experiments/<resultsFolder>/times/estimators/``.
                Useful when the caller manages its own output directory layout
                (e.g. per-player directories under ``Data/``).
        """

        self.execution_skills = execution_skills
        self.num_rationality_levels = num_rationality_levels
        self.domain_name = domain_name
        self._times_base_dir = times_base_dir
        self.method_type = "JT-QRE"  # (same thing as JEEDS)

        # Baseline names used for MAP and EES estimates
        base = f"{self.method_type}-{{}}-{self.num_execution_skills}-{self.num_rationality_levels}"
        base_names = [base.format("MAP"), base.format("EES")]

        self.label = self._build_label(given_prior, min_lambda, other_args)

        # Append the label to both estimator names so downstream metrics are tagged consistently
        base_names[0] += self.label
        base_names[1] += self.label
        self.names = base_names

        # Storage for posterior traces of execution and planning skills (per estimator name)
        self.estimates_execution_skills = dict()
        self.estimates_rationality_levels = dict()

        self.estimates_execution_skills = {n: [] for n in self.names}
        self.estimates_rationality_levels = {n: [] for n in self.names}

        self.rationality_levels = self._init_rationality_levels(
            domain_name=domain_name,
            num_rationality_levels=self.num_rationality_levels,
            min_lambda=min_lambda,
            other_args=other_args,
        )

        self.current_probs = self._init_joint_prior(
            execution_skills=self.execution_skills,
            given_prior=given_prior,
            other_args=other_args,
        )

        # Keep a history of the posterior over time (first entry is the prior)
        self.probs_history = [self.current_probs.tolist()]

    @property
    def num_execution_skills(self):
        """Number of execution-skill hypotheses currently tracked."""
        return len(self.execution_skills)

    def _init_joint_prior(self, execution_skills, given_prior, other_args):
        """Initialize the joint prior P(x, p) over execution and planning skills."""

        if given_prior:
            # Skew-normal prior over execution skills; assume planning skills are uniformly likely
            x_probs = skewnorm.pdf(
                execution_skills,
                a=other_args["givenPrior"][0],
                loc=other_args["givenPrior"][1],
                scale=other_args["givenPrior"][2],
            )
            x_probs = x_probs.reshape(x_probs.shape[0], 1)

            probs = x_probs.copy()
            probs = probs.reshape(probs.shape[0], 1)

            # Fill the rest of the columns with the same value (uniform across planning skills)
            for _ in range(self.num_rationality_levels - 1):
                probs = np.hstack((probs, x_probs))

            probs /= np.sum(probs)
            return probs

        # Uniform distribution across all execution/planning combinations
        probs = np.ndarray(shape=(self.num_execution_skills, self.num_rationality_levels))
        probs.fill(1.0 / (self.num_execution_skills * self.num_rationality_levels * 1.0))
        return probs

    @staticmethod
    def _init_rationality_levels(domain_name, num_rationality_levels, min_lambda, other_args):
        """ Create the grid of planning skill hypotheses (lambda values).
            Note that np.logspace returns values evenly spaced in log-10 space; i.e.,
            10^start, 10^(start+delta), etc.
        """
        if domain_name == "1d":
            return np.logspace(-3, 2, num_rationality_levels)  # 0.001 to 100
        if domain_name in ["2d", "2d-multi", "sequentialDarts", "billiards"]:
            return np.logspace(-3, 1.5, num_rationality_levels)
        if domain_name in ["baseball", "baseball-multi"]:
            if min_lambda:
                return np.logspace(other_args["minLambda"], 3.6, num_rationality_levels)
            return np.logspace(-3, 3.6, num_rationality_levels)
        if domain_name == "hockey-multi":
            return np.round(np.logspace(0, 3.6, num_rationality_levels), 4)
        return None

    def get_estimator_name(self):
        return self.names

    def mid_reset(self):
        """Reset collected estimates but keep the current posterior intact."""

        self.estimates_execution_skills = {n: [] for n in self.names}
        self.estimates_rationality_levels = {n: [] for n in self.names}

        self.probs_history = []

    def reset(self):
        """Fully reset the estimator, including the joint posterior distribution."""

        self.estimates_execution_skills = {n: [] for n in self.names}
        self.estimates_rationality_levels = {n: [] for n in self.names}

        # reset probs
        self.current_probs.fill(1.0 / (self.num_execution_skills * self.num_rationality_levels * 1.0))

        self.probs_history = []
        self.probs_history.append(self.current_probs.tolist())

    def _init_observation_info(self, spaces, action, other_args):
        """Extract common per-observation inputs and normalize types.

        Returns
        -------
        results_folder : str
        tag : str
        delta : any
            Resolution parameter from `spaces`.
        current_score : any | None
            Only set for sequential darts.
        action : np.ndarray
        """

        results_folder = other_args["resultsFolder"]
        tag = other_args["tag"]
        delta = spaces.delta

        current_score = None
        if self.domain_name == "sequentialDarts":
            current_score = other_args["currentScore"]

        action = np.array(action)
        return results_folder, tag, delta, current_score, action

    def _update_target_distributions(self, rng, spaces, state):
        # Update target distributions for each execution hypothesis when the domain requires it.
        # Baseball/hockey pass all required information directly through otherArgs.
        for execution_skill in self.execution_skills:
            if self.domain_name == "2d-multi":
                spaces.updateSpace(rng, [[[execution_skill, execution_skill]], [0.0]], state)
            else:
                spaces.updateSpace(rng, [execution_skill], state)

    def _compute_pdfs_and_evs(self, spaces, action, current_score, delta, other_args):
        # Containers reused across hypotheses to avoid recomputation later during the update stage
        pdfs_per_execution_skill = {}
        evs_per_execution_skill = {}

        for exec_skill_index, execution_skill in enumerate(self.execution_skills):

            # Get the corresponding xskill level hypothesis at the given index
            if self.domain_name in ["2d-multi", "baseball-multi", "hockey-multi"]:
                # Multidimensional setups use a tuple-like key built from both dimensions
                key = spaces.get_key([execution_skill, execution_skill], r=0.0)
            else:
                key = execution_skill

            if self.domain_name in ["1d", "2d", "2d-multi"]:

                # Look up precomputed convolutions for this execution skill and state index
                space = spaces.convolutionsPerXskill[key][other_args["s"]]
                evs = space["all_vs"].flatten()

                listed_targets = spaces.listedTargets  # Shared grid of possible targets for darts-like domains

                if self.domain_name == "1d":
                    pdfs = norm.pdf([action] * len(listed_targets), loc=listed_targets,
                                    scale=[execution_skill] * len(listed_targets))
                else:  # 2D
                    pdfs = computePDF(x=action, means=listed_targets,
                                      covs=np.array([spaces.convolutionsPerXskill[key]["cov"]] * len(listed_targets)))

            elif self.domain_name == "sequentialDarts":
                # Sequential darts spaces are stored per execution skill and indexed by score
                space = spaces.spacesPerXskill[execution_skill]
                evs = space.flatEVsPerState[current_score]

                # Each execution skill has its own covariance matrix per target
                pdfs = computePDF(x=action, means=spaces.possibleTargets,
                                  covs=spaces.allCovsGivenXskillDomainTargets[exec_skill_index])

            elif self.domain_name == "billiards":

                # Sampling
                pdfs, evs = getPDFsAndEVsBilliardsSampling(spaces, execution_skill, action, other_args)

                '''
                # New way of computing EVs (focal ev)
                pdfs,evs = getPDFsAndEVsBilliardsFocalEV(spaces,self.xskills,action,otherArgs)
                '''

            elif self.domain_name in ["baseball", "baseball-multi"]:

                # Baseball/hockey feed EVs through infoPerRow rather than precomputed convolutions
                evs = other_args["infoPerRow"]["evsPerXskill"][key].flatten()
                cov = spaces.all_covs[key]

                pdfs = computePDF(x=action, means=spaces.possibleTargetsFeet,
                                  covs=np.array([cov] * len(spaces.possibleTargetsFeet)))

                # code.interact("JTM...", local=dict(globals(), **locals()))

            elif self.domain_name in ["hockey-multi"]:

                evs = other_args["infoPerRow"]["evsPerXskill"][key].flatten()
                cov = spaces.all_covs[key]

                pdfs = computePDF(x=action, means=spaces.possibleTargets,
                                  covs=np.array([cov] * len(spaces.possibleTargets)))

                '''
                df = 3 
                # # pdfs = multivariate_t.pdf(action,loc=spaces.possibleTargets,shape=np.array([cov]*len(spaces.possibleTargets)),df=df)

                pdfs = np.array([
                    multivariate_t.pdf(action,loc=mean,shape=cov,df=df)
                    for mean in spaces.possibleTargets])
                '''

            else:
                raise ValueError(
                    f"Unsupported domain_name '{self.domain_name}' encountered in _compute_pdfs_and_evs."
                )

            if self.domain_name == "1d":

                # Normalize pdfs - (since modeling as continuous but using discrete set of targets)
                # If sum != 0 to prevent NANs
                # if np.sum(pdfs) != 0.0:
                # 	pdfs /= np.sum(pdfs)
                np.multiply(pdfs, np.square(delta))

            elif self.domain_name == "hockey-multi":
                # pdfs = np.multiply(pdfs,np.square(delta[0]*delta[1]))
                pdfs /= np.sum(pdfs)

            elif self.domain_name in ["2d", "2d-multi", "sequentialDarts", "baseball", "baseball-multi"]:
                # Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
                # This is because depending on the xskill/resolution combination, the pdf of
                # a given xskill may not show up in any of the resolution buckets
                # causing then the pdfs not adding up to 1
                # (example: xskill of 1.0 & resolution > 1.0)
                # If the resolution is less than the xskill, the xskill distribution can be fully captured
                # by the resolution thus avoiding problems.
                pdfs = np.multiply(pdfs, np.square(delta))

            # Save info
            pdfs_per_execution_skill[execution_skill] = pdfs
            evs_per_execution_skill[execution_skill] = evs

        return pdfs_per_execution_skill, evs_per_execution_skill

    def _perform_update(self, pdfs_per_execution_skill, evs_per_execution_skill):

        for exec_skill_index, execution_skill in enumerate(self.execution_skills):

            pdfs = pdfs_per_execution_skill[execution_skill]
            evs = evs_per_execution_skill[execution_skill]

            # If resulting posterior distribution for possible targets
            # given xskill hyp & executed action results in all 0's
            # Means there's no way you'll be of this xskill
            # So no need to update probs, can remain 0.0
            if np.sum(pdfs) == 0.0 or np.isnan(np.sum(pdfs)):
                self.current_probs[exec_skill_index] = [0.0] * self.num_rationality_levels
                print(f"skipping (pdfs sum = 0) - x hyp: {execution_skill}")
                continue

            # For each rationality level hypothesis
            for rationality_index, rationality_level in enumerate(self.rationality_levels):
                # Applying norm trick (subtract b to avoid overflow in the exponential for large EV*lambda values)
                b = np.max(evs * rationality_level)
                expev = np.exp(evs * rationality_level - b)
                sum_exponents = np.sum(expev)

                # JT Update
                summultexps = np.sum(np.multiply(expev, np.copy(pdfs)))

                upd = summultexps / sum_exponents

                # Update probs
                self.current_probs[exec_skill_index][rationality_index] *= upd

    def _update_estimates_from_posterior(self):
        # MAP estimate - Get index of maximum prob - returns flat index
        mi = np.argmax(self.current_probs)

        # "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
        xmi, pmi = np.unravel_index(mi, self.current_probs.shape)

        self.estimates_execution_skills[self.names[0]].append(self.execution_skills[xmi])
        self.estimates_rationality_levels[self.names[0]].append(self.rationality_levels[pmi])

        # Get EES & EPS Estimate (expected execution/planning skill under current posterior)
        ees = 0.0
        eps = 0.0

        for xi in range(self.num_execution_skills):
            for pi in range(self.num_rationality_levels):
                ees += self.execution_skills[xi] * self.current_probs[xi][pi]
                eps += self.rationality_levels[pi] * self.current_probs[xi][pi]

        self.estimates_execution_skills[self.names[1]].append(ees)
        self.estimates_rationality_levels[self.names[1]].append(eps)

        return ees, eps

    def _print_and_log_data(self, results_folder, tag, expected_exec_skill, expected_rationality_level, total_time_est):
        print("JT-QRE")
        print("EES: ", expected_exec_skill, "\t\t MAP: ", self.estimates_execution_skills[self.names[0]][-1])
        print("EPS: ", expected_rationality_level, "\t\t MAP: ", self.estimates_rationality_levels[self.names[0]][-1],
              "\n")

        # Ensure output folders exist
        if self._times_base_dir is not None:
            base_dir = self._times_base_dir
        else:
            base_dir = os.path.join("Experiments", results_folder)
        times_dir = os.path.join(base_dir, "times")
        estimators_dir = os.path.join(times_dir, "estimators")
        os.makedirs(estimators_dir, exist_ok=True)

        # Append timing data
        out_path = os.path.join(estimators_dir, f"JT-QRE-Times-{tag}")
        with open(out_path, "a") as file:
            file.write(f"{total_time_est}\n")

    def add_observation(self, rng, spaces, state, action, **other_args):
        """Update posterior distributions given a single observed action.

        Parameters
        ----------
        rng : np.random.Generator
                Random generator used by spaces when updating target sets.
        spaces : object
                Domain-specific structure that exposes target grids, convolutions, and covariances.
        state : any
                Current environment state (used by space updates).
        action : array-like
                Executed action to condition on.
        other_args : dict
                Additional metadata required by different domains (e.g., currentScore, resultsFolder).
            """

        results_folder, tag, delta, current_score, action = self._init_observation_info(
            spaces=spaces,
            action=action,
            other_args=other_args,
        )

        start_time_est = time.perf_counter()

        if "baseball" not in self.domain_name and "hockey" not in self.domain_name:
            self._update_target_distributions(rng, spaces, state)

        pdfs_per_execution_skill, evs_per_execution_skill = self._compute_pdfs_and_evs(
            spaces,
            action,
            current_score,
            delta,
            other_args
        )

        self._perform_update(pdfs_per_execution_skill, evs_per_execution_skill)

        # Normalize to ensure a valid probability distribution
        self.current_probs /= np.sum(self.current_probs)

        # Update the probs history
        self.probs_history.append(self.current_probs.tolist())

        # Get expected execution skill and rationality under current posterior
        expected_exec_skill, expected_rationality_level = self._update_estimates_from_posterior()

        end_time_est = time.perf_counter()
        total_time_est = end_time_est - start_time_est

        self._print_and_log_data(results_folder, tag, expected_exec_skill, expected_rationality_level, total_time_est)

    def get_results(self):
        results = dict()

        for n in self.names:
            results[f"{n}-pSkills"] = self.estimates_rationality_levels[n]
            results[f"{n}-xSkills"] = self.estimates_execution_skills[n]

        if self.label != "":
            results[f"{self.method_type}{self.label}-allProbs"] = self.probs_history
        else:
            results[f"{self.method_type}-allProbs"] = self.probs_history
        return results

                # code.interact("JTM...", local=dict(globals(), **locals()))

class NonJointMethodQRE:

    def __init__(self, xskills, numPskills, domainName, givenPrior=False, minLambda=False, otherArgs=None):

        self.xskills = xskills
        self.numXskills = len(xskills)
        self.numPskills = numPskills

        self.domainName = domainName

        self.methodType = "NJT-QRE"

        baseNames = [f"{self.methodType}-MAP-{self.numXskills}-{self.numPskills}",
                     f"{self.methodType}-EES-{self.numXskills}-{self.numPskills}"]

        if givenPrior and minLambda:
            self.label = f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}-MinLambda{otherArgs['minLambda']}"
        elif givenPrior:
            self.label = f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}"
        elif minLambda:
            self.label = f"-MinLambda-{otherArgs['minLambda']}"
        else:
            self.label = ""

        baseNames[0] += self.label
        baseNames[1] += self.label
        self.names = baseNames

        self.estimatesXskills = dict()
        self.estimatesPskills = dict()

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesPskills[n] = []

        if domainName == "1d":
            # 0.001 to 100
            self.pskills = np.logspace(-3, 2, self.numPskills)
        elif domainName in ["2d", "sequentialDarts", "billiards"]:
            self.pskills = np.logspace(-3, 1.5, self.numPskills)
        elif domainName in ["baseball"]:
            if minLambda:
                self.pskills = np.logspace(otherArgs["minLambda"], 3.6, self.numPskills)
            else:
                self.pskills = np.logspace(-3, 3.6, self.numPskills)

        self.probsPskills = np.ndarray(shape=(self.numPskills, 1))
        self.probsPskills.fill(1.0 / (self.numPskills * 1.0))

        # Initializing the array (init prior distribution)
        if givenPrior:
            # Get "skewed" dist for the different xskill hyps
            xProbs = skewnorm.pdf(self.xskills, a=otherArgs["givenPrior"][0], loc=otherArgs["givenPrior"][1],
                                  scale=otherArgs["givenPrior"][2])
            xProbs = xProbs.reshape(xProbs.shape[0], 1)

            self.probsXskills = xProbs.copy()
            # Normalize
            self.probsXskills /= np.sum(self.probsXskills)
            # Reshape
            self.probsXskills = self.probsXskills.reshape(self.probsXskills.shape[0], 1)

        else:
            self.probsXskills = np.ndarray(shape=(self.numXskills, 1))
            self.probsXskills.fill(1.0 / (self.numXskills * 1.0))

        # To save the initial probs - uniform distribution for all
        self.allProbsXskills = [self.probsXskills.tolist()]
        self.allProbsPskills = [self.probsPskills.tolist()]

        # code.interact("NJTM init...", local=dict(globals(), **locals()))

    def getEstimatorName(self):
        return self.names

    def midReset(self):

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesPskills[n] = []

        self.allProbsXskills = []
        self.allProbsPskills = []

    def reset(self):

        for n in self.names:
            self.estimatesPskills[n] = []
            self.estimatesXskills[n] = []

        # Reset probs
        self.probsXskills.fill(1.0 / (self.numXskills * 1.0))
        self.probsPskills.fill(1.0 / (self.numPskills * 1.0))

        self.allProbsXskills = []
        self.allProbsPskills = []

        # To save the initial probs - uniform distribution for all
        self.allProbsXskills.append(self.probsXskills.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())

    # @profile
    def addObservation(self, spaces, action, **otherArgs):

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialize info
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        tag = otherArgs["tag"]
        delta = spaces.delta
        resultsFolder = otherArgs["resultsFolder"]

        if self.domainName == "sequentialDarts":
            currentScore = otherArgs["currentScore"]

        action = np.array(action)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        PDFsPerXskill = {}
        EVsPerXskill = {}

        startTimeEst = time.perf_counter()

        ##################################################################################################################################
        # ESTIMATING EXECUTION SKILL
        ##################################################################################################################################

        # For each execution skill hyps
        for xi in range(len(self.xskills)):

            # Get the corresponding xskill level hypothesis at the given index
            x = self.xskills[xi]

            ###########################################################################################
            # Compute PDFs and EVs
            ###########################################################################################

            if self.domainName == "1d" or self.domainName == "2d":

                space = spaces.convolutionsPerXskill[x][otherArgs["s"]]
                evs = space["all_vs"].flatten()
                listedTargets = spaces.listedTargets

                if self.domainName == "1d":
                    pdfs = scipy.stats.norm.pdf([action] * len(listedTargets), loc=listedTargets,
                                                scale=[x] * len(listedTargets))

                else:  # 2D
                    pdfs = computePDF(x=action, means=listedTargets,
                                      covs=np.array([spaces.convolutionsPerXskill[x]["cov"]] * len(listedTargets)))


            elif self.domainName == "sequentialDarts":
                space = spaces.spacesPerXskill[x]
                evs = space.flatEVsPerState[currentScore]

                pdfs = computePDF(x=action, means=spaces.possibleTargets,
                                  covs=np.array([spaces.all_covs[xi]] * len(spaces.possibleTargets)))

            elif self.domainName == "billiards":

                # Sampling
                pdfs, evs = getPDFsAndEVsBilliardsSampling(spaces, x, action, otherArgs)

                '''
                # New way of computing EVs (focal ev)
                pdfs,evs = getPDFsAndEVsBilliardsFocalEV(spaces,xskills,action,otherArgs)
                '''

            elif self.domainName == "baseball":

                evs = otherArgs["infoPerRow"]["evsPerXskill"][x].flatten()

                pdfs = computePDF(x=action, means=spaces.possibleTargetsFeet,
                                  covs=[spaces.all_covs[xi]] * len(spaces.possibleTargetsFeet))

            if self.domainName == "1d":
                # norm pdfs - (since modeling as continuous but using discrete set of targets)
                # If sum != 0 to prevent NANs
                # if np.sum(pdfs) != 0.0:
                # 	pdfs /= np.sum(pdfs)
                pdfs = np.multiply(pdfs, np.square(delta))

            elif self.domainName in ["2d", "sequentialDarts", "baseball"]:
                pdfs = np.multiply(pdfs, np.square(delta))

            # Store in order to reuse later on when updating pskills probs
            PDFsPerXskill[x] = pdfs
            EVsPerXskill[x] = evs

            ###########################################################################################

            v3 = []

            # For each planning skill hyp
            for pi in range(len(self.pskills)):
                # Get the corresponding pskill level at the given index
                p = self.pskills[pi]

                # Create copy of EVs
                evsCP = np.copy(evs)

                # To be used for exp normalization trick - find maxEV and * by p
                # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
                # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
                b = np.max(evsCP * p)

                # With normalization trick
                expev = np.exp(evsCP * p - b)

                # exps = V1
                sumexp = np.sum(expev)

                V2 = expev / sumexp

                # Non-JT Update

                mult = np.multiply(V2, pdfs)

                # upd = v2
                upd = np.sum(mult)

                # Update probs pskill
                v3.append(self.probsPskills[pi] * upd)

            # Update probs xskill
            self.probsXskills[xi] *= np.sum(v3)

        ##################################################################################################################################

        ##################################################################################################################################
        # ESTIMATING PLANNING SKILL
        ##################################################################################################################################

        # For each pskill hyp
        for pi in range(len(self.pskills)):

            # Get the corresponding pskill level at the given index
            p = self.pskills[pi]

            v3 = []

            # For each xskill hyp
            for xi in range(len(self.xskills)):
                # Get the corresponding xskill level hypothesis at the given index
                x = self.xskills[xi]

                pdfs = PDFsPerXskill[x]
                evs = EVsPerXskill[x]

                # Create copy of EVs
                evsC = np.copy(evs)

                # To be used for exp normalization trick - find maxEV and * by p
                # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
                # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
                b = np.max(evsCP * p)

                # With normalization trick
                expev = np.exp(evsCP * p - b)

                # exps = V1
                sumexp = np.sum(expev)

                V2 = expev / sumexp

                # Non-JT Update

                mult = np.multiply(V2, pdfs)

                # upd = v2
                upd = np.sum(mult)

                # Update probs pskill
                v3.append(self.probsXskills[xi] * upd)

            # Update probs pskill
            self.probsPskills[pi] *= np.sum(v3)

        ##################################################################################################################################

        # Once done updating the different probabilities, proceed to get estimates

        # Normalize
        self.probsXskills /= np.sum(self.probsXskills)
        self.probsPskills /= np.sum(self.probsPskills)

        self.allProbsXskills.append(self.probsXskills.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())

        # Get estimate. Uses MAP estimate
        # Get index of maximum prob
        xmi = np.argmax(self.probsXskills)
        pmi = np.argmax(self.probsPskills)

        # code.interact("...", local=dict(globals(), **locals()))

        self.estimatesXskills[self.names[0]].append(self.xskills[xmi])
        self.estimatesPskills[self.names[0]].append(self.pskills[pmi])

        # Get EES Estimate
        ees = 0.0
        # print "probs xskills: "
        for xi in range(len(self.xskills)):
            # print "x: " + str(self.xskills[xi]) + "->" + str(self.probsXskills[pi])
            # [0] in order to get number out of array
            # To avoid problems when saving results to file since results in an array within an array
            ees += self.xskills[xi] * self.probsXskills[xi][0]

        # Get EPS Estimate
        eps = 0.0
        # print "probs pskills: "
        for pi in range(len(self.pskills)):
            # print "p: " + str(self.pskills[pi]) + "-> " + str(self.probsPskills[pi][0])
            # [0] in order to get number out of array
            # To avoid problems when saving results to file since results in an array within an array
            eps += self.pskills[pi] * self.probsPskills[pi][0]

        self.estimatesXskills[self.names[1]].append(ees)
        self.estimatesPskills[self.names[1]].append(eps)

        endTimeEst = time.perf_counter()
        totalTimeEst = endTimeEst - startTimeEst

        # print("NJT-QRE")
        # print("EES: ", ees, "\t\t MAP: ", self.estimatesXskills[self.names[0]][-1])
        # print("EPS: ", eps, "\t\t MAP: ", self.estimatesPskills[self.names[0]][-1], "\n")
        # code.interact("...", local=dict(globals(), **locals()))

        folder = resultsFolder

        # If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep)

        # If the folder doesn't exist already, create it
        if not os.path.exists(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators"):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators")

        with open(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators" + os.path.sep + "NonJT-QRE-Times-" + tag,
                "a") as file:
            file.write(str(totalTimeEst) + "\n")

    def getResults(self):
        results = dict()

        for n in self.names:
            results[f"{n}-pSkills"] = self.estimatesPskills[n]
            results[f"{n}-xSkills"] = self.estimatesXskills[n]

        if self.label != "":
            results[f"{self.methodType}{self.label}-xSkills-allProbs"] = self.allProbsXskills
            results[f"{self.methodType}{self.label}-pSkills-allProbs"] = self.allProbsPskills
        else:
            results[f"{self.methodType}-xSkills-allProbs"] = self.allProbsXskills
            results[f"{self.methodType}-pSkills-allProbs"] = self.allProbsPskills

        return results


class QREMethod_Multi:

    def __init__(self, xskills, numPskills, rhos, domainName, givenPrior=False, minLambda=False, otherArgs=None):

        self.xskills = xskills
        self.rhos = rhos

        self.numPskills = numPskills

        self.domainName = domainName

        self.methodType = "QRE-Multi"
        self.subTypes = ["-JT"]  # ,"-NJT"]

        baseNames = [f"{self.methodType}{self.subTypes[0]}-MAP",
                     f"{self.methodType}{self.subTypes[0]}-EES"]
        # f"{self.methodType}{self.subTypes[1]}-MAP",
        # f"{self.methodType}{self.subTypes[1]}-EES"]

        self.label = ""

        for each in self.xskills:
            temp = f"-{len(each)}"
            self.label += temp

        for temp in [f"-{len(self.rhos)}", f"-{self.numPskills}"]:
            self.label += temp

        if givenPrior and minLambda:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}-MinLambda{otherArgs['minLambda']}"
        elif givenPrior:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}"
        elif minLambda:
            self.label += f"-MinLambda-{otherArgs['minLambda']}"

        for i in range(len(baseNames)):
            baseNames[i] += self.label

        self.names = baseNames

        self.estimatesXskills = dict()
        self.estimatesRhos = dict()
        self.estimatesPskills = dict()

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesPskills[n] = []

        if domainName in ["2d-multi", "sequentialDarts", "billiards"]:
            self.pskills = np.round(np.logspace(-3, 1.5, self.numPskills), 4)
        elif domainName in ["baseball"]:
            if minLambda:
                self.pskills = np.round(np.logspace(otherArgs["minLambda"], 3.6, self.numPskills), 4)
            else:
                self.pskills = np.round(np.logspace(-3, 3.6, self.numPskills), 4)
        elif domainName in ["hockey-multi"]:
            if minLambda:
                self.pskills = np.round(np.logspace(otherArgs["minLambda"], 3.6, self.numPskills), 4)
            else:
                self.pskills = np.round(np.logspace(0, 3.6, self.numPskills), 4)

        # FOR TESTING
        # self.xskills = [self.xskills[0],self.xskills[1],self.xskills[0],self.xskills[1]]

        if len(self.xskills) == 1:
            self.allParams = list(product(self.xskills[0]))
        else:
            self.allParams = list(product(self.xskills[0], self.xskills[1]))

            if len(self.xskills) >= 2:

                for di in range(2, len(self.xskills)):
                    self.allParams = list(product(self.allParams, self.xskills[di]))

        self.allParams = list(product(self.allParams, self.rhos))
        self.allParams = list(product(self.allParams, self.pskills))

        for ii in range(len(self.allParams)):
            self.allParams[ii] = eval(str(self.allParams[ii]).replace(")", "").replace("(", ""))

        # code.interact("...", local=dict(globals(), **locals()))

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NEED TO UPDATE TO MULTIPLE DIMENSIONS
        # (and replicate for NJT)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Initializing the array (init prior distribution)
        if givenPrior:
            pass

            # # Get "skewed" dist for the different xskill hyps
            # xProbs = skewnorm.pdf(self.xskills,a=otherArgs["givenPrior"][0],loc=otherArgs["givenPrior"][1],scale=otherArgs["givenPrior"][2])
            # xProbs = xProbs.reshape(xProbs.shape[0],1)

            # self.probs = xProbs.copy()
            # self.probs = self.probs.reshape(self.probs.shape[0],1)

            # # Fill the rest of the columns with the same value
            # # Assumption: All lambdas are equally likely (uniform distribution)
            # for i in range(self.numPskills-1):
            # 	self.probs = np.hstack((self.probs,xProbs))

            # self.probs /= np.sum(self.probs)

        else:
            temp = ()
            mult = 1.0

            for each in self.xskills:
                temp += (len(each),)
                mult *= len(each)

            ########################################
            # FOR JOINT
            ########################################

            temp += (len(self.rhos), self.numPskills)
            self.probs = np.ndarray(shape=temp)
            self.probs.fill(1.0 / (mult * len(self.rhos) * self.numPskills * 1.0))

            ########################################

            ########################################
            # FOR NON-JOINT
            ########################################

            '''
            self.probsXskills = {}
            self.dims = {}

            d = 0

            for each in self.xskills:
                size = len(each)
                self.probsXskills[d] = np.ndarray(shape=(size,1))
                self.probsXskills[d].fill(1.0/(size*1.0))
                self.dims[d] = size
                d += 1

            self.probsPskills = np.ndarray(shape=(self.numPskills,1))
            self.probsPskills.fill(1.0/(self.numPskills*1.0))

            self.probsRhos = np.ndarray(shape=(len(self.rhos),1))
            self.probsRhos.fill(1.0/(len(self.rhos)*1.0))
            '''

            ########################################

        # To save the initial probs - uniform distribution for all
        self.allProbs = [self.probs.tolist()]

        '''
        self.allProbsXskills = {}

        for d in self.probsXskills:
            self.allProbsXskills[d] = [self.probsXskills[d].tolist()]

        self.allProbsRhos = [self.probsRhos.tolist()]
        self.allProbsPskills = [self.probsPskills.tolist()]
        '''

        self.indexes = {}
        self.indexes["rhos"] = {}
        self.indexes["ps"] = {}

        for d in range(len(self.xskills)):

            xs = self.xskills[d]
            key = f"x-{d}"
            self.indexes[key] = {}

            for i in range(len(xs)):
                if xs[i] not in self.indexes[key]:
                    self.indexes[key][xs[i]] = i

        for i in range(len(self.rhos)):
            if self.rhos[i] not in self.indexes["rhos"]:
                self.indexes["rhos"][self.rhos[i]] = i

        for i in range(len(self.pskills)):
            if self.pskills[i] not in self.indexes["ps"]:
                self.indexes["ps"][self.pskills[i]] = i

        # code.interact("init...", local=dict(globals(), **locals()))

    def getEstimatorName(self):
        return self.names

    def reset(self):

        for n in self.names:
            self.estimatesPskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesXskills[n] = []

        mult = 1.0
        for each in self.xskills:
            mult *= len(each)

        # Reset probs
        self.probs.fill(1.0 / (mult * len(self.rhos) * self.numPskills * 1.0))

        self.allProbs = []
        self.allProbs.append(self.probs.tolist())

        '''
        for d in self.probsXskills:
            size = self.dims[d]
            self.probsXskills[d] = np.ndarray(shape=(size,1))
            self.probsXskills[d].fill(1.0/(size*1.0))

        self.probsRhos.fill(1.0/(len(self.rhos)*1.0))
        self.probsPskills.fill(1.0/(self.numPskills*1.0))


        self.allProbsXskills = {}

        # To save the initial probs - uniform distribution for all
        for d in self.probsXskills:
            self.allProbsXskills[d] = [self.probsXskills[d].tolist()]


        self.allProbsRhos = []
        self.allProbsPskills = []

        self.allProbsRhos.append(self.probsRhos.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())
        '''

    def addObservation(self, rng, spaces, state, action, **otherArgs):

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialize info
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        resultsFolder = otherArgs["resultsFolder"]
        tag = otherArgs["tag"]
        delta = spaces.delta

        action = np.array(action)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        startTimeEst = time.perf_counter()

        if spaces.mode != "normal":
            for each in self.allParams:
                if type(each) == list:
                    spaces.updateSpace(rng, [[each[:-2]], [each[-2]]], state)
                else:
                    spaces.updateSpace(rng, each, state)

        # code.interact("...", local=dict(globals(), **locals()))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Compute PDFs and EVs
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        PDFsPerXskill = {}
        EVsPerXskill = {}

        for each in self.allParams:

            key = "|".join(map(str, each[:-1]))

            if self.domainName in ["2d-multi"]:
                space = spaces.convolutionsPerXskill[key][otherArgs["s"]]
                listedTargets = spaces.listedTargets

                # TODO: check this out later. this was a hotfix using AI. before this line used "x" where it now uses "key"

                pdfs = computePDF(x=action, means=listedTargets,
                                  covs=np.array([spaces.convolutionsPerXskill[key]["cov"]] * len(listedTargets)))

                # Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
                # This is because depending on the xskill/resolution combination, the pdf of
                # a given xskill may not show up in any of the resolution buckets
                # causing then the pdfs not adding up to 1
                # (example: xskill of 1.0 & resolution > 1.0)
                # If the resolution is less than the xskill, the xskill distribution can be fully captured
                # by the resolution thus avoiding problems.
                pdfs = np.multiply(pdfs, np.square(delta))

            # Save info
            PDFsPerXskill[key] = pdfs
            EVsPerXskill[key] = space["all_vs"].flatten()

            # code.interact("...", local=dict(globals(), **locals()))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Perform Joint Update
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for each in self.allParams:

            xs, r, p = each[:-2], each[-2], each[-1]

            iis = []
            key = ""

            for d in range(len(xs)):
                iis.append(self.indexes[f"x-{d}"][xs[d]])
                key += f"{xs[d]}|"

            indexR = self.indexes["rhos"][r]
            key += f"{r}"
            iis.append(indexR)

            indexP = self.indexes["ps"][p]
            iis.append(indexP)

            iis = tuple(iis)

            pdfs = PDFsPerXskill[key]
            evs = EVsPerXskill[key]

            # If resulting posterior distribution for possible targets
            # given xskill hyp & executed action results in all 0's
            # Means there's no way you'll be of this xskill
            # So no need to update probs, can remain 0.0
            if np.sum(pdfs) == 0.0:
                self.probs[iis[:-1]] = [0.0] * self.numPskills
                continue

            # Create copy of EVs
            evsC = np.copy(evs)

            # To be used for exp normalization trick - find maxEV and * by p
            # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
            # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            b = np.max(evsC * p)

            # With normalization trick
            expev = np.exp(evsC * p - b)

            sumexp = np.sum(expev)

            # JT Update
            summultexps = np.sum(np.multiply(expev, np.copy(pdfs)))

            # Update probs
            self.probs[iis] *= (summultexps / sumexp)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Perform Non-Joint Update
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        '''
        # Compute marginals from joint distribution)
        marginals = scipy.stats.contingency.margins(self.probs)
        updXS, updRhos, updPskills = marginals[:-2], marginals[-2],marginals[-1]

        for d in range(len(updXS)):
            upd = updXS[d].reshape(self.dims[d],1)

            self.probsXskills[d] *= upd

        start_time_est = time.perf_counter()

        updRhos = updRhos.reshape(len(self.rhos),1)
        updPskills = updPskills.reshape(len(self.pskills),1)

        self.probsRhos *= updRhos
        self.probsPskills *= updPskills
        '''

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Normalize
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.probs /= np.sum(self.probs)
        self.allProbs.append(self.probs.tolist())

        '''
        for d in self.probsXskills:
            self.probsXskills[d] /= np.sum(self.probsXskills[d])

        self.probsRhos /= np.sum(self.probsRhos)
        self.probsPskills /= np.sum(self.probsPskills)

        self._print_and_log_data(results_folder, tag, expected_exec_skill, expected_rationality_level, total_time_est)

        for d in self.probsXskills:
            self.allProbsXskills[d].append(self.probsXskills[d].tolist())

        self.allProbsRhos.append(self.probsRhos.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())
        '''

        # code.interact("after norm ", local=dict(globals(), **locals()))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Get estimates - For Joint
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # MAP estimate - Get index of maximum prob - returns flat index
        mi = np.argmax(self.probs)

        # "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
        iis = np.unravel_index(mi, self.probs.shape)

        xsi, ri, pi = iis[:-2], iis[-2], iis[-1]

        estimates = []

        for d in range(len(xsi)):
            estimates.append(self.xskills[d][xsi[d]])

        self.estimatesXskills[self.names[0]].append(estimates)
        self.estimatesRhos[self.names[0]].append(self.rhos[ri])
        self.estimatesPskills[self.names[0]].append(self.pskills[pi])
        # code.interact("...", local=dict(globals(), **locals()))

        # Get Expected Estimate
        ees = [0.0] * len(xsi)
        ers = 0.0
        eps = 0.0

        for each in self.allParams:

            xs, r, p = each[:-2], each[-2], each[-1]

            iis = []

            for d in range(len(xs)):
                iis.append(self.indexes[f"x-{d}"][xs[d]])

            indexR = self.indexes["rhos"][r]
            iis.append(indexR)

            indexP = self.indexes["ps"][p]
            iis.append(indexP)

            iis = tuple(iis)

            for d in range(len(xs)):
                ees[d] += xs[d] * self.probs[iis]

            ers += r * self.probs[iis]
            eps += p * self.probs[iis]

        self.estimatesXskills[self.names[1]].append(ees)
        self.estimatesRhos[self.names[1]].append(ers)
        self.estimatesPskills[self.names[1]].append(eps)

        '''
        print("JT-QRE-Multi")
        print(f"EES:{ees}  |  MAP: {self.estimatesXskills[self.names[0]][-1]}")
        print(f"ERS:{ers}  |  MAP: {self.estimatesRhos[self.names[0]][-1]}")
        print(f"EPS:{eps}  |  MAP: {self.estimatesPskills[self.names[0]][-1]}\n")
        # code.interact("...", local=dict(globals(), **locals()))
        '''

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Get estimates - For Non-Joint
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        '''
        # MAP estimate - Get index of maximum prob

        temp = []
        for d in self.probsXskills:
            xmi = np.argmax(self.probsXskills[d])
            temp.append(self.xskills[d][xmi])
        self.estimatesXskills[self.names[2]].append(temp)


        rmi = np.argmax(self.probsRhos)
        pmi = np.argmax(self.probsPskills)

        self.estimatesRhos[self.names[2]].append(self.rhos[rmi])
        self.estimatesPskills[self.names[2]].append(self.pskills[pmi])


        # Get Expected Estimate
        ees = [0.0]*len(temp)

        for d in range(len(self.xskills)):

            xs = self.xskills[d]

            for i in range(len(xs)):
                ees[d] += xs[d] * self.probsXskills[d][i][0]


        ers = 0.0
        for ri in range(len(self.rhos)):
            ers += self.rhos[pi] * self.probsRhos[ri][0]

        eps = 0.0
        for pi in range(len(self.pskills)):
            eps += self.pskills[pi] * self.probsPskills[pi][0]


        self.estimatesXskills[self.names[3]].append(ees)
        self.estimatesRhos[self.names[3]].append(ers) 
        self.estimatesPskills[self.names[3]].append(eps) 

        # print("NJT-QRE-Multi")
        # print(f"EES:{ees}  |  MAP: {self.estimatesXskills[self.names[2]][-1]}")
        # print(f"ERS:{ers}  |  MAP: {self.estimatesRhos[self.names[2]][-1]}")
        # print(f"EPS:{eps}  |  MAP: {self.estimatesPskills[self.names[2]][-1]}\n")
        # code.interact("...", local=dict(globals(), **locals()))
        '''

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if spaces.mode != "normal":
            for each in self.allParams:
                spaces.deleteSpace(list(each[:-1]), state)

        endTimeEst = time.perf_counter()
        totalTimeEst = endTimeEst - startTimeEst

        folder = resultsFolder

        # If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep)

        # If the folder doesn't exist already, create it
        if not os.path.exists(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators"):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators")

        with open(
                "Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators" + os.path.sep + "JT-QRE-Times-" + tag,
                "a") as file:
            file.write(str(totalTimeEst) + "\n")

    def getResults(self):

        results = dict()

        for n in self.names:
            results[f"{n}-xSkills"] = self.estimatesXskills[n]
            results[f"{n}-rhos"] = self.estimatesRhos[n]
            results[f"{n}-pSkills"] = self.estimatesPskills[n]

        # For Joint
        results[f"{self.methodType}{self.label}-allProbs"] = self.allProbs

        # For Non-Joint
        # results[f"{self.methodType}{self.label}-xSkills-allProbs"] = self.allProbsXskills
        # results[f"{self.methodType}{self.label}-rhos-allProbs"] = self.allProbsRhos
        # results[f"{self.methodType}{self.label}-pSkills-allProbs"] = self.allProbsPskills

        # Other info
        results[f"{self.methodType}{self.label}-xskills"] = self.xskills
        results[f"{self.methodType}{self.label}-rhos"] = self.rhos.tolist()
        results[f"{self.methodType}{self.label}-pskills"] = self.pskills.tolist()

        # code.interact("...", local=dict(globals(), **locals()))

        return results


'''
class NonJointMethodQRE_Multi:

    def __init__(self,xskills,numPskills,domainName,givenPrior=False,minLambda=False,otherArgs=None):
        self.xskills = xskills
        self.rhos = rhos

        self.numPskills = numPskills

        self.domainName = domainName

        self.methodType = "NJT-QRE-Multi"

        baseNames = [f"{self.methodType}-MAP",
                    f"{self.methodType}-EES"]

        self.label = ""

        for each in self.xskills:
            temp = f"-{len(each)}"
            self.label += temp

        for temp in [f"-{len(self.rhos)}",f"-{self.numPskills}"]:
            self.label += temp

        if givenPrior and minLambda:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}-MinLambda{otherArgs['minLambda']}"
        elif givenPrior:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}"
        elif minLambda:
            self.label += f"-MinLambda-{otherArgs['minLambda']}"

        baseNames[0] += self.label
        baseNames[1] += self.label
        self.names = baseNames


        self.estimatesXskills = dict()
        self.estimatesRhos = dict()
        self.estimatesPskills = dict()

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesPskills[n] = []


        if domainName in ["2d", "sequentialDarts","billiards"]:
            self.pskills = np.logspace(-3,1.5,self.numPskills)
        elif domainName in ["baseball"]:
            if minLambda:
                self.pskills = np.logspace(otherArgs["minLambda"],3.6,self.numPskills)
            else:
                self.pskills = np.logspace(-3,3.6,self.numPskills)


        self.probsPskills = np.ndarray(shape=(self.numPskills,1))
        self.probsPskills.fill(1.0/(self.numPskills*1.0))

        self.probsRhos = np.ndarray(shape=(len(self.rhos),1))
        self.probsRhos.fill(1.0/(len(self.rhos)*1.0))


        # FOR TESTING
        # self.xskills = [self.xskills[0],self.xskills[1],self.xskills[0],self.xskills[1]]

        if len(self.xskills) == 1:
            self.allParams = list(product(self.xskills[0]))
        else:
            self.allParams = list(product(self.xskills[0],self.xskills[1]))

            if len(self.xskills) >= 2:

                for di in range(2,len(self.xskills)):
                    self.allParams = list(product(self.allParams,self.xskills[di]))

        self.allParams = list(product(self.allParams,self.rhos))
        self.allParams = list(product(self.allParams,self.pskills))

        for ii in range(len(self.allParams)):
            self.allParams[ii] = eval(str(self.allParams[ii]).replace(")","").replace("(",""))


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NEED TO UPDATE TO MULTIPLE DIMENSIONS
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Initializing the array (init prior distribution)
        if givenPrior:
            pass
            # # Get "skewed" dist for the different xskill hyps
            # xProbs = skewnorm.pdf(self.xskills,a=otherArgs["givenPrior"][0],loc=otherArgs["givenPrior"][1],scale=otherArgs["givenPrior"][2])
            # xProbs = xProbs.reshape(xProbs.shape[0],1)

            # self.probsXskills = xProbs.copy()
            # # Normalize
            # self.probsXskills /= np.sum(self.probsXskills)
            # # Reshape
            # self.probsXskills = self.probsXskills.reshape(self.probsXskills.shape[0],1)

        else:

            self.probsXskills = {}
            self.dims = {}

            d = 1

            for each in self.xskills:
                size = len(each)
                self.probsXskills[d] = np.ndarray(shape=(size,1))
                self.probsXskills[d].fill(1.0/(size*1.0))
                self.dims[d] = size
                d += 1


        # To save the initial probs - uniform distribution for all
        self.allProbsXskills = {}

        for d in self.probsXskills:
            self.allProbsXskills[d] = [self.probsXskills[d].tolist()]

        self.allProbsRhos = [self.probsRhos.tolist()]
        self.allProbsPskills = [self.probsPskills.tolist()]

        # code.interact("NJTM init...", local=dict(globals(), **locals()))


    def getEstimatorName(self):
        return self.names


    def midReset(self):

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesPskills[n] = []

        self.allProbsXskills = {}
        self.allProbsRhos = []
        self.allProbsPskills = []


    def reset(self):

        for n in self.names:
            self.estimatesPskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesXskills[n] = []

        # Reset probs
        for d in self.probsXskills:
            self.probsXskills[d].fill(1.0/(self.dims[d]*1.0))

        self.probsRhos.fill(1.0/(len(self.rhos)*1.0))
        self.probsPskills.fill(1.0/(self.numPskills*1.0))

        self.allProbsXskills = {}
        self.allProbsRhos = []
        self.allProbsPskills = []

        # To save the initial probs - uniform distribution for all
        for d in self.probsXskills:
            self.allProbsXskills[d] = [self.probsXskills[d].tolist()]

        self.allProbsRhos.append(self.probsRhos.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())


    # @profile
    def addObservation(self,spaces,action,**otherArgs):

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialize info
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        tag = otherArgs["tag"]
        delta = spaces.delta
        resultsFolder = otherArgs["resultsFolder"]

        if self.domainName == "sequentialDarts":
            currentScore = otherArgs["currentScore"]

        action = np.array(action)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        PDFsPerXskill = {}
        EVsPerXskill = {}


        startTimeEst = time.perf_counter()

        ##################################################################################################################################
        # ESTIMATING EXECUTION SKILL
        ##################################################################################################################################

        tempUpd = {}


        for each in self.allParams:

            xs, r, p = each[:-2], each[-2], each[-1]

            iis = []
            key = ""

            for d in range(len(xs)):
                iis.append(self.indexes[f"x-{d}"][xs[d]])
                key += f"{xs[d]}|"

            indexR = self.indexes["rhos"][r]
            key += f"{r}"
            iis.append(indexR)

            indexP = self.indexes["ps"][p]
            iis.append(indexP)

            iis = tuple(iis)



            ###########################################################################################
            # Compute PDFs and EVs
            ###########################################################################################

            if self.domainName in ["2d-multi"]:

                space = spaces.convolutionsPerXskill[key][otherArgs["s"]]
                evs = space["all_vs"].flatten()

                listedTargets = spaces.listedTargets
                pdfs = computePDF(x=action,means=listedTargets,covs=np.array([spaces.convolutionsPerXskill[x]["cov"]]*len(listedTargets)))

                pdfs = np.multiply(pdfs,np.square(delta))


            # Store in order to reuse later on when updating pskills probs
            PDFsPerXskill[key] = pdfs
            EVsPerXskill[key] = evs

            ###########################################################################################

            v3 = []

            # For each planning skill hyp
            for pi in range(len(self.pskills)):

                # Get the corresponding pskill level at the given index
                p = self.pskills[pi]


                # Create copy of EVs 
                evsCP = np.copy(evs)

                # To be used for exp normalization trick - find maxEV and * by p
                # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
                # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
                b = np.max(evsCP*p)

                # With normalization trick
                expev = np.exp(evsCP*p-b) 

                # exps = V1
                sumexp = np.sum(expev)

                V2 = expev/sumexp

                # Non-JT Update 

                mult = np.multiply(V2,pdfs)

                # upd = v2
                upd = np.sum(mult)

                # Update probs pskill
                v3.append(self.probsPskills[pi] * upd)


            # Update probs xskill
            self.probsXskills[xi] *= np.sum(v3)

        ##################################################################################################################################


        ##################################################################################################################################
        # ESTIMATING PLANNING SKILL
        ##################################################################################################################################

        # For each pskill hyp
        for pi in range(len(self.pskills)):

            # Get the corresponding pskill level at the given index
            p = self.pskills[pi]


            v3 = []

            # For each xskill hyp
            for xi in range(len(self.xskills)):

                # Get the corresponding xskill level hypothesis at the given index
                x = self.xskills[xi]

                pdfs = PDFsPerXskill[x]
                evs = EVsPerXskill[x]

                # Create copy of EVs 
                evsC = np.copy(evs)

                # To be used for exp normalization trick - find maxEV and * by p
                # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
                # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
                b = np.max(evsCP*p)

                # With normalization trick
                expev = np.exp(evsCP*p-b) 

                sumexp = np.sum(expev)

                # Non-JT Update 

                V2 = expev/sumexp

                mult = np.multiply(V2,pdfs)

                upd = np.sum(mult)

                # Update probs pskill
                v3.append(self.probsXskills[xi]*upd)


            # Update probs pskill
            self.probsPskills[pi] *= np.sum(v3)

        ##################################################################################################################################

        # Once done updating the different probabilities, proceed to get estimates

        # Normalize
        self.probsXskills /= np.sum(self.probsXskills)
        self.probsPskills /= np.sum(self.probsPskills)


        self.allProbsXskills.append(self.probsXskills.tolist())
        self.allProbsPskills.append(self.probsPskills.tolist())


        # Get estimate. Uses MAP estimate
        # Get index of maximum prob
        xmi = np.argmax(self.probsXskills)
        pmi = np.argmax(self.probsPskills)

        # code.interact("...", local=dict(globals(), **locals()))

        self.estimatesXskills[self.names[0]].append(self.xskills[xmi])
        self.estimatesPskills[self.names[0]].append(self.pskills[pmi])


        #Get EES Estimate
        ees = 0.0
        #print "probs xskills: "
        for xi in range(len(self.xskills)):
            # print "x: " + str(self.xskills[xi]) + "->" + str(self.probsXskills[pi])
            # [0] in order to get number out of array
            # To avoid problems when saving results to file since results in an array within an array
            ees += self.xskills[xi] * self.probsXskills[xi][0]


        #Get EPS Estimate
        eps = 0.0
        #print "probs pskills: "
        for pi in range(len(self.pskills)):
            # print "p: " + str(self.pskills[pi]) + "-> " + str(self.probsPskills[pi][0])
            # [0] in order to get number out of array
            # To avoid problems when saving results to file since results in an array within an array
            eps += self.pskills[pi] * self.probsPskills[pi][0]


        self.estimatesXskills[self.names[1]].append(ees)
        self.estimatesPskills[self.names[1]].append(eps)        

        endTimeEst = time.perf_counter()
        totalTimeEst = endTimeEst-startTimeEst

        # print("NJT-QRE")
        # print("EES: ", ees, "\t\t MAP: ", self.estimatesXskills[self.names[0]][-1])
        # print("ERS: ", ers, "\t\t MAP: ", self.estimatesRhos[self.names[0]][-1], "\n")
        # print("EPS: ", eps, "\t\t MAP: ", self.estimatesPskills[self.names[0]][-1], "\n")
        # code.interact("...", local=dict(globals(), **locals()))

        folder = resultsFolder

        #If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep)

        #If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators"):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators")

        with open("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators" + os.path.sep + "NonJT-QRE-Times-"+ tag, "a") as file:
            file.write(str(totalTimeEst) + "\n")


    def getResults(self):
        results = dict()

        for n in self.names:
            results[f"{n}-pSkills"] = self.estimatesPskills[n]
            results[f"{n}-rhos"] = self.estimatesRhos[n]
            results[f"{n}-xSkills"] = self.estimatesXskills[n]

        results[f"{self.methodType}{self.label}-xSkills-allProbs"] = self.allProbsXskills
        results[f"{self.methodType}{self.label}-rhos-allProbs"] = self.allProbsRhos.tolist()
        results[f"{self.methodType}{self.label}-pSkills-allProbs"] = self.allProbsPskills.tolist()

        return results
'''

