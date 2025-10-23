import numpy as np
import scipy
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as mcolors
import matplotlib
import itertools
import sys
import time 
import os
import code
from matplotlib.cm import ScalarMappable

"""One-dimensional darts environment utilities."""

# This module defines the 1D darts environment and helper utilities for the
# broader skill estimation framework. Several other modules import values from
# here directly, so backwards-compatible aliases are preserved where needed.

BOARD_LIMIT = 10
# ``m`` is kept for compatibility with older code that accesses ``darts.m``
# directly. Prefer ``BOARD_LIMIT`` within this module for clarity.
m = BOARD_LIMIT
 
def get_domain_name():
    """Return the identifier used by the framework for this domain."""

    return "1d"

def draw_noise_sample(rng, noise_std_dev):
    """Sample a single noise value for an executed throw."""

    return rng.normal(0.0, noise_std_dev)

def plot_states_with_agent_details(states, game_summaries, results_folder):
    """Plot each state alongside detailed agent performance information."""

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_cycle = list(colors.keys())

    for state_index in range(len(states)):
        figure = plt.figure()
        axis = plt.subplot(111)

        plot_reward_profile(states[state_index])

        for agent_index in range(len(game_summaries)):
            agent_name = game_summaries[agent_index][0].split("-X")[0].split("Agent")[0]
            agent_name_parts = game_summaries[agent_index][0].split("-")
            noise_level = agent_name_parts[1].replace("X","")

            plt.plot(
                game_summaries[agent_index][1][state_index],
                game_summaries[agent_index][2][state_index],
                label=agent_name + "-" + agent_name_parts[-1],
                linestyle='None',
                marker="s",
                color=color_cycle[agent_index],
            )
            plt.plot(
                game_summaries[agent_index][3][state_index],
                game_summaries[agent_index][4][state_index],
                linestyle='None',
                marker="P",
                color=color_cycle[agent_index],
            )
            plt.plot(
                game_summaries[agent_index][3][state_index],
                np.mean(game_summaries[agent_index][5][state_index]),
                linestyle='None',
                marker="*",
                color=color_cycle[agent_index],
            )

        plot_expected_values(states[state_index], float(noise_level), color="k")

        # Shrink current axis by 10% to make room for the legend.
        axis_box = axis.get_position()
        axis.set_position([axis_box.x0, axis_box.y0, axis_box.width * 0.9, axis_box.height])

        # Put a legend to the right of the current axis.
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 14})

        plt.title("xSkill: " + str(agent_name_parts[-2]))
        plt.savefig(
            results_folder + os.path.sep + agent_name_parts[-2] + "-state" + str(state_index) + ".png",
            bbox_inches='tight',
        )

        plt.clf()
        plt.close(figure)

def plot_reward_profile(state_boundaries, color="black"):
    """Plot the piecewise constant reward profile defined by ``state_boundaries``."""

    is_low_region = True

    x_points = [-BOARD_LIMIT]
    y_points = [0.0]

    for boundary in state_boundaries:
        x_points.extend([boundary, boundary])
        if is_low_region:
            y_points.extend([0.0, 1.0])
            is_low_region = False
        else:
            y_points.extend([1.0, 0.0])
            is_low_region = True

    x_points.extend([state_boundaries[-1], BOARD_LIMIT])
    y_points.extend([0.0, 0.0])
    plt.plot(x_points, y_points, label="Reward", color=color)

def get_rewards_for_plot(state_boundaries):
    """Return the reward level for each alternating region in the state."""

    rewards = []
    is_low_region = True

    for boundary in state_boundaries:
        if is_low_region:
            rewards.append(1)
            is_low_region = False
        else:
            rewards.append(0)
            is_low_region = True

    return rewards

def plot_expected_values(state_boundaries, noise_std_dev, color='r', label=None):
    """Plot the expected value curve for a given noise level."""

    expected_values, actions = compute_expected_value_curve(state_boundaries, noise_std_dev)

    if label is not None:
        plt.plot(actions, expected_values, color, label=label)
    else:
        plt.plot(actions, expected_values, color)

def get_reward_for_action(rng, state_boundaries, action):
    """Return the reward earned by executing ``action`` in the given state."""

    is_low_region = True
    for boundary in state_boundaries:
        if action < boundary:
            break
        is_low_region = not is_low_region

    return 0.0 if is_low_region else 1.0

def find_state_interval(state_boundaries, action):
    """Return the index pair describing the interval that contains ``action``."""

    for index in range(len(state_boundaries)):
        if index != len(state_boundaries) - 1:
            if action >= float(state_boundaries[index]) and action <= float(state_boundaries[index + 1]):
                return index, index + 1

    if action >= -BOARD_LIMIT and action <= float(state_boundaries[0]):
        return -BOARD_LIMIT, 0

    if action >= float(state_boundaries[len(state_boundaries) - 1]) and action <= BOARD_LIMIT:
        return len(state_boundaries) - 1, BOARD_LIMIT
    
def is_action_within_interval(state_boundaries, action, left_index, right_index):
    """Return True when ``action`` lies inside the provided interval."""

    if action >= float(state_boundaries[left_index]) and action <= float(state_boundaries[right_index]):
        return True

    if left_index == -BOARD_LIMIT:
        return action >= -BOARD_LIMIT and action <= float(state_boundaries[right_index])

    if right_index == BOARD_LIMIT:
        return action >= float(state_boundaries[left_index]) and action <= BOARD_LIMIT

    return False

def calculate_random_reward(state_boundaries):
    """Return the reward of uniformly sampling actions on the board."""

    total_success_width = 0.0
    for index in range(0, len(state_boundaries), 2):
        total_success_width += abs(state_boundaries[index] - state_boundaries[index + 1])

    board_width = 2 * BOARD_LIMIT
    return total_success_width / board_width

def wrap_action_within_bounds(action):
    """Wrap an action so it remains within the circular dart board."""

    while action > BOARD_LIMIT:
        action = action - 2 * BOARD_LIMIT
    while action < -BOARD_LIMIT:
        action = action + 2 * BOARD_LIMIT
    return action

def sample_noisy_action(rng, state_boundaries, noise_std_dev, action, noise_model=None):
    """Return the executed action after applying Gaussian execution noise."""

    if noise_model is None:
        noise = draw_noise_sample(rng, noise_std_dev)
    else:
        noise = noise_model

    noisy_action = action + noise
    noisy_action = wrap_action_within_bounds(noisy_action)

    return noisy_action

# Note: the actions passed into this function are assumed to already lie in [-BOARD_LIMIT, BOARD_LIMIT]
def calculate_wrapped_action_difference(action_1, action_2):
    """Return the difference between two actions on the wrapped board."""

    difference = action_1 - action_2
    if difference > BOARD_LIMIT:
        difference -= 2 * BOARD_LIMIT
    if difference < -BOARD_LIMIT:
        difference += 2 * BOARD_LIMIT

    return difference

def sample_single_rollout(rng, state_boundaries, noise_std_dev, action):
    """Sample the reward obtained by executing ``action`` once."""

    noisy_action = sample_noisy_action(rng, state_boundaries, noise_std_dev, action)
    return get_reward_for_action(rng, state_boundaries, noisy_action)


def estimate_value_with_samples(rng, state_boundaries, noise_std_dev, num_samples, action):
    """Estimate the expected value of ``action`` via Monte Carlo sampling."""

    total_reward = 0.0
    for _ in range(num_samples):
        total_reward += sample_single_rollout(rng, state_boundaries, noise_std_dev, action)
    return total_reward / float(num_samples)

def compute_expected_value_curve(state_boundaries, noise_std_dev, delta=1e-2):
    """Return expected values for all actions at the provided noise level."""

    num_points = int(6 * BOARD_LIMIT / delta)
    action_grid = np.linspace(-3 * BOARD_LIMIT, 3 * BOARD_LIMIT, num_points)

    state_values = [
        get_reward_for_action(None, state_boundaries, wrap_action_within_bounds(action))
        for action in action_grid
    ]

    error_distribution = stats.norm(loc=0, scale=noise_std_dev)
    error_pmf = error_distribution.pdf(action_grid) * delta

    convolved_values = np.convolve(state_values, error_pmf, 'same')

    left = int(num_points / 3)
    right = int(2 * left)

    return convolved_values[left:right], action_grid[left:right]

def generate_random_states(rng, low, high, count, min_width=0.0):
    """Generate ``count`` random dart board states."""

    _states = []

    for _ in range(count):
        num_regions = rng.integers(low, high) * 2

        region_count = 0
        boundaries = []

        while region_count < num_regions:
            candidate = rng.uniform(-BOARD_LIMIT, BOARD_LIMIT)
            valid_point = True
            for boundary in boundaries:
                if abs(candidate - boundary) < min_width:
                    valid_point = False
                    break
            if valid_point:
                boundaries.append(candidate)
                region_count += 1

        boundaries = np.sort(boundaries)
        _states.append(boundaries.tolist())

    return _states

def get_optimal_action_and_value(rng, state_boundaries, noise_std_dev, resolution):
    """Return the action that maximizes expected reward and its value."""

    expected_values, actions = compute_expected_value_curve(state_boundaries, noise_std_dev, resolution)
    best_index = np.argmax(expected_values)
    return actions[best_index], expected_values[best_index]


def get_expected_values_and_optimal_action(rng, state_boundaries, noise_std_dev, resolution):
    """Return the EV curve and the optimal action/value pair."""

    expected_values, actions = compute_expected_value_curve(state_boundaries, noise_std_dev, resolution)
    best_index = np.argmax(expected_values)
    return expected_values, actions[best_index], expected_values[best_index]

def verify_expected_value_convolution(rng, xskills, state_boundaries):
    """Compare Monte Carlo estimates with the convolution-based EV."""

    for noise_std_dev in xskills:
        expected_values, actions = compute_expected_value_curve(state_boundaries, noise_std_dev)

        best_index = np.argmax(expected_values)
        best_action = actions[best_index]

        print(f"xskill: {noise_std_dev}")
        print(f"action: {best_action}")

        reward_sum = 0.0
        num_rollouts = 10_000
        for _ in range(num_rollouts):
            noisy_action = sample_noisy_action(rng, state_boundaries, noise_std_dev, best_action)
            reward = get_reward_for_action(rng, state_boundaries, noisy_action)
            reward_sum += reward

        average_reward = reward_sum / num_rollouts
        print(f"avg: {average_reward} | EV: {expected_values[best_index]}\n")

    # code.interact("...", local=dict(globals(), **locals()))

def simulate_board_hits(rng, xskills, state_boundaries, num_trials, aim=""):
    """Estimate hit rates for different execution skills."""

    hit_percentages = []

    print(f"Aiming at: {aim}")

    for noise_std_dev in xskills:
        expected_values, actions = compute_expected_value_curve(state_boundaries, noise_std_dev)

        if aim == "optimal":
            best_action = actions[np.argmax(expected_values)]
        else:
            best_action = 0.0

        hits = 0.0

        for _ in range(int(num_trials)):
            noisy_action = sample_noisy_action(rng, state_boundaries, noise_std_dev, best_action)

            if not (noisy_action < -BOARD_LIMIT or noisy_action > BOARD_LIMIT):
                hits += 1.0

        percent_hit = (hits / num_trials) * 100.0
        hit_percentages.append(percent_hit)

        print("xSkill: ", noise_std_dev, "| \tTotal Hits: ", hits, " out of ", num_trials, "-> ", percent_hit, "%")

    return hit_percentages


# ---------------------------------------------------------------------------
# Backwards compatibility aliases
# ---------------------------------------------------------------------------
getDomainName = get_domain_name
getNoiseModel = draw_noise_sample
plot_state_allInfo = plot_states_with_agent_details
plot_state = plot_reward_profile
getRewardsForPlot = get_rewards_for_plot
plot_ev = plot_expected_values
get_v = get_reward_for_action
findRegion = find_state_interval
checkIfActionInRegion = is_action_within_interval
get_rand_reward = calculate_random_reward
wrap_action = wrap_action_within_bounds
sample_action = sample_noisy_action
actionDiff = calculate_wrapped_action_difference
sample_1 = sample_single_rollout
sample_N = estimate_value_with_samples
convolve_ev = compute_expected_value_curve
get_N_states = generate_random_states
get_target = get_optimal_action_and_value
get_all_targets = get_expected_values_and_optimal_action
verifyConvolveEV = verify_expected_value_convolution
testHits = simulate_board_hits


if __name__ == '__main__':

    ##################################################
    # PARAMETERS FOR PLOTS
    ##################################################

    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams.update({'legend.fontsize': 14})
    plt.rcParams["axes.titleweight"] = "bold"

    #plt.rcParams["font.weight"] = "bold"
    #plt.rcParams["axes.labelweight"] = "bold"

    ##################################################


    # seed = np.random.randint(0,1000000,1)
    seed = 10
    rng = np.random.default_rng(seed)

    folder = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
    
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder+"Plots-States"):
        os.mkdir(folder+"Plots-States")

    if not os.path.exists(folder+f"Plots-States{os.sep}Wrap"):
        os.mkdir(folder+f"Plots-States{os.sep}Wrap")


    # xskills = np.round(np.linspace(0.5,4.5,num=5),4)
    xskills = np.round(np.linspace(0.5,20.0,num=10),4)

    colors = ["tab:orange","tab:green","tab:red","tab:purple","tab:pink"]
    colors += ["tab:brown","tab:gray","tab:olive","tab:cyan","b"]
    # colors = cm.rainbow(np.linspace(0,1,len(xskills)))

    num_states = 5

    states = generate_random_states(rng, 3, 5, num_states, 0.5)

    num_trials = 10_000

    for state_index in range(1):  # (num_states):
        # verify_expected_value_convolution(rng, xskills, states[state_index])

        hit_percentages = simulate_board_hits(rng, xskills, states[state_index], num_trials, aim="optimal")
        hit_percentages = simulate_board_hits(rng, xskills, states[state_index], num_trials, aim="middle")

    
    code.interact("...", local=dict(globals(), **locals()))


    for noise_std_dev in xskills:
        expected_values, actions = compute_expected_value_curve(states[0], noise_std_dev)

        best_index = np.argmax(expected_values)
        best_action = actions[best_index]

        print(f"xskill: {noise_std_dev}")
        print(f"action: {best_action}")

        reward_sum = 0.0
        num_rollouts = 10_000
        for _ in range(num_rollouts):
            noisy_action = sample_noisy_action(rng, states[0], noise_std_dev, best_action)
            reward = get_reward_for_action(rng, states[0], noisy_action)
            reward_sum += reward

        average_reward = reward_sum / num_rollouts
        print(f"avg: {average_reward} | EV: {expected_values[best_index]}")

        code.interact("...", local=dict(globals(), **locals()))



    for state_index in range(num_states):

        # '''
        fig = plt.figure()
        ax = plt.gca()
        #ax = plt.subplot2grid((5,2), (0, 0))

        plot_reward_profile(states[state_index], color="tab:blue")

        plt.margins(0.10)
        ax.autoscale(True)

        plt.ylim(-0.1,1.1)

        for j in range(len(xskills)):
            plot_expected_values(states[state_index], xskills[j], color=colors[j], label=xskills[j])


        plt.legend()

        plt.tight_layout()
        plt.savefig(folder+f"Plots-States{os.sep}Wrap{os.sep}1D-state" + str(state_index)+f"-EVs-WrapTrue.png")
       
        plt.close()
        plt.clf()
        # '''


        # '''
        for j in range(len(xskills)):

            if not os.path.exists(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}"):
                os.mkdir(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}")

            fig = plt.figure()
            ax = plt.gca()

            expected_values, actions = compute_expected_value_curve(states[state_index], xskills[j])

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(min(expected_values), max(expected_values))
            sm = ScalarMappable(norm = norm, cmap = cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm,ax=ax)
            cbar.ax.set_title("EVs")

            for ii in range(len(actions)):
                plt.vlines(x=actions[ii], ymin=0, ymax=expected_values[ii], colors=cmap(norm(expected_values[ii])))
            
            plot_reward_profile(states[state_index], color="tab:blue")
            plt.title(f"State: {state_index} | Xskill: {xskills[j]}")
            plt.tight_layout()
            plt.savefig(f"{folder}Plots-States{os.sep}Wrap{os.sep}xskill{xskills[j]}{os.sep}1D-state{state_index}.png")
            
            plt.close()
            plt.clf()

        # '''


        '''
        from plotly import graph_objs as go
        import plotly as py

        rewards = get_rewards_for_plot(states[state_index])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=states[state_index], y=rewards, mode='markers', marker_size=20,
        ))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, 
                         zeroline=True, zerolinecolor='black', zerolinewidth=3,
                         showticklabels=False)
        #fig.update_layout(height=200, plot_bgcolor='white')
        fig.layout.update(height=200, plot_bgcolor='white')

        # Save plotly
        unique_url = py.offline.plot(fig, filename= "1D-state" + str(state_index)+".html", auto_open=False)

        '''

        '''
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(8, 3))
        ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])



        rewards = get_rewards_for_plot(states[state_index])
        # print(rewards)

        colors = ["w"]
        for r in rewards:
            if r == 0:
                colors.append("w")
            else:
                colors.append("tab:blue")

        # The second example illustrates the use of a ListedColormap, a
        # BoundaryNorm, and extended ends to show the "over" and "under"
        # value colors.
        cmap = matplotlib.colors.ListedColormap(colors)
        # cmap.set_over('0.25')
        # cmap.set_under('0.75')

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = [-BOARD_LIMIT] + states[state_index] + [BOARD_LIMIT]
        print(bounds)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                        norm=norm,
                                        # to use 'extend', you must
                                        # specify two extra boundaries:
                                        boundaries= bounds ,
                                        # extend='both',
                                        ticks=bounds,  # optional
                                        spacing='proportional',
                                        orientation='horizontal')
        # cb2.set_label('Discrete intervals, some other units')

        # clb = plt.colorbar()
        # cb2.set_label('1', labelpad=-82, y=-500.0, rotation=0)

        plt.text(-0.02,1,"1")
        plt.text(-0.02,0,"0")

        degrees = 45 #90
        plt.xticks(rotation=degrees)


        #plt.show()
        plt.savefig("1D-state" + str(state_index)+".png")
        '''
