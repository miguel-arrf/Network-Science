import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.generators import random_graphs
import altair as alt

alt.renderers.enable('altair_viewer')

# Number of individuals
N = 400

# Number of simulations
simulations = 1

# Number of generations
# generations = 400 * 10
generations = 110

# Number of pairs
# m = 2 * N
m = 1200

# Number of cooperative interactions
cooperative_interactions = 0
all_interactions = 0

# Cost
c = 0.1

# Benefit
b = 1

# Strategy values range
minimum_strategy = -5
maximum_strategy = 6

# Image score values range
minimum_image_score = -5
maximum_image_score = 5


class Agent:
    def __init__(self, strategy):
        self.image_score = 0
        self.payoff = 0
        self.strategy = strategy

    def increase_image_score(self):
        if self.image_score < maximum_image_score:
            self.image_score = self.image_score + 1

    def decrease_image_score(self):
        if self.image_score > minimum_image_score:
            self.image_score = self.image_score - 1


def play(to_draw=False, barabasi=True):
    global cooperative_interactions
    global all_interactions

    last_generations = []

    agents = []
    if barabasi:
        network = random_graphs.barabasi_albert_graph(N, 6)
    else:
        network = random_graphs.watts_strogatz_graph(N, 5, 0.1)
    coalitions = []
    cooperation_ratio = []

    lines_graph = {-5: [], -4: [], -3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    coalitions_size = []

    for agent in range(N):
        agent_to_add = Agent(random.randint(minimum_strategy, maximum_strategy))
        agents.append(agent_to_add)
        network.nodes[agent]['agent'] = agent_to_add

    for simulation in range(simulations):

        for generation in range(generations):

            if generation % 100 == 0:
                print("<----->")
                print("Generation: ", generation)
                print("Coalitions: ", coalitions)
                for coalition in coalitions:
                    if len(coalition) == 0:
                        print("empty coalition!")

            coalitions_size.append(len(coalitions))

            for pair in range(m):

                donor, recipient = find_players(agents)

                # Let's know if recipient is on the coalition of donor or if it is a neighbor!
                if is_neighbor(donor, recipient, network) or is_peer(donor, recipient, coalitions):
                    if donor.strategy <= recipient.image_score:
                        recipient.payoff += b
                        donor.payoff -= c
                        cooperative_interactions += 1
                        donor.increase_image_score()
                    else:
                        donor.decrease_image_score()

                else:
                    donor.decrease_image_score()

                all_interactions += 1

            # Now each individual either joins a coalition or removes itself from one
            for agent in agents:
                agent_coalitions = [coalition for coalition in coalitions if agent in coalition]

                if len(agent_coalitions) == 1:
                    if not anyNeighbor_in_agents_coalition(agent, agent_coalitions, network):
                        agent_coalitions[0].remove(agent)
                        for coalition in coalitions:
                            if len(coalition) == 0:
                                coalitions.remove(coalition)

                elif len(agent_coalitions) > 1:
                    print("GET WRECK!")

                # Updating the agent_coalition vector, to see if we still are in any!
                agent_coalitions = [coalition for coalition in coalitions if agent in coalition]

                # We may be in some coalition!
                if agent_was_worst_in_neighbors(agent, network):
                    best_independent_agent, best_independent_score = agent_wants_best_independent_neighbor(agent,
                                                                                                           network,
                                                                                                           coalitions)
                    best_coalition, best_coalition_score = agent_wants_best_coalition(agent, network, coalitions)

                    if best_independent_score is not None and best_coalition is not None:

                        if len(agent_coalitions) != 0:
                            agent_coalitions[0].remove(agent)
                            for coalition in coalitions:
                                if len(coalition) == 0:
                                    coalitions.remove(coalition)

                        if best_coalition_score > best_independent_score:
                            best_coalition.append(agent)
                        else:
                            coalitions.append([best_independent_agent, agent])

                    elif best_independent_agent is None and best_coalition is not None:
                        # No independent agent, but coalitions exist -> Let's join!
                        if len(agent_coalitions) != 0:
                            agent_coalitions[0].remove(agent)
                            for coalition in coalitions:
                                if len(coalition) == 0:
                                    coalitions.remove(coalition)

                        best_coalition.append(agent)
                    elif best_independent_agent is not None and best_coalition is None:
                        # There is independent agent but no coalition exists -> Let's create!
                        if len(agent_coalitions) != 0:
                            agent_coalitions[0].remove(agent)
                            for coalition in coalitions:
                                if len(coalition) == 0:
                                    coalitions.remove(coalition)

                        coalitions.append([best_independent_agent, agent])

                agent.strategy = agent_wants_best_neighbour_strategy_without_old_values(agent, network)

            '''
            # Now each individual imitates a given strategy
            for agent in range(N):
                network.nodes[agent]['oldValue'] = network.nodes[agent]['agent'].strategy


            for agent in agents:
                agent.strategy = agent_wants_best_neighbour_strategy(agent, network)
            '''

            for agent in agents:
                agent.payoff = 0
                agent.image_score = 0

            minus_5 = len([x for x in agents if x.strategy == (-5)]) / N
            minus_4 = len([x for x in agents if x.strategy == (-4)]) / N
            minus_3 = len([x for x in agents if x.strategy == (-3)]) / N
            minus_2 = len([x for x in agents if x.strategy == (-2)]) / N
            minus_1 = len([x for x in agents if x.strategy == (-1)]) / N
            minus_0 = len([x for x in agents if x.strategy == 0]) / N
            plus_1 = len([x for x in agents if x.strategy == 1]) / N
            plus_2 = len([x for x in agents if x.strategy == 2]) / N
            plus_3 = len([x for x in agents if x.strategy == 3]) / N
            plus_4 = len([x for x in agents if x.strategy == 4]) / N
            plus_5 = len([x for x in agents if x.strategy == 5]) / N
            plus_6 = len([x for x in agents if x.strategy == 6]) / N

            lines_graph[-5].append(minus_5)
            lines_graph[-4].append(minus_4)
            lines_graph[-3].append(minus_3)
            lines_graph[-2].append(minus_2)
            lines_graph[-1].append(minus_1)
            lines_graph[0].append(minus_0)
            lines_graph[1].append(plus_1)
            lines_graph[2].append(plus_2)
            lines_graph[3].append(plus_3)
            lines_graph[4].append(plus_4)
            lines_graph[5].append(plus_5)
            lines_graph[6].append(plus_6)

            cooperation_ratio.append(cooperative_interactions / all_interactions)


            if generation % 100 == 0 and generation > 0 and to_draw:
                plt.plot(coalitions_size, color="#2F80ED")
                plt.savefig('coalitions-at-{}.pdf'.format(generation))
                plt.show()

                plt.plot(cooperation_ratio, color="#2F80ED")
                plt.savefig('coperation-at-{}.pdf'.format(generation))
                plt.show()

                x = range(generation + 1)
                plt.plot(x, lines_graph[-5], label="-5", color="#2F80ED")
                plt.plot(x, lines_graph[-4], label="-4", color="#2F80ED")
                plt.plot(x, lines_graph[-3], label="-3", color="#2F80ED")
                plt.plot(x, lines_graph[-2], label="-2", color="#2F80ED")
                plt.plot(x, lines_graph[-1], label="-1", color="#2F80ED")
                plt.plot(x, lines_graph[0], label="0", color="#2F80ED")
                plt.plot(x, lines_graph[1], 'r--', label="1", dash_capstyle="round")
                plt.plot(x, lines_graph[2], 'r--', label="2", dash_capstyle="round")
                plt.plot(x, lines_graph[3], 'r--', label="3", dash_capstyle="round")
                plt.plot(x, lines_graph[4], 'r--', label="4", dash_capstyle="round")
                plt.plot(x, lines_graph[5], 'r--', label="5", dash_capstyle="round")
                plt.plot(x, lines_graph[6], 'r--', label="6", dash_capstyle="round")

                '''
                cooperative_strategies = len(lines_graph[-5]) + len(lines_graph[-4]) + len(lines_graph[-3]) + \
                                         len(lines_graph[-2]) + len(lines_graph[-1]) + len(lines_graph[0])

                non_cooperative_strategies = len(lines_graph[1]) + len(lines_graph[2]) + len(lines_graph[3]) + \
                                         len(lines_graph[4]) + len(lines_graph[5]) + len(lines_graph[6])

                print("cooperative: ", cooperative_strategies)
                print("non-cooperative: ", non_cooperative_strategies)

                print("sum: ", cooperative_strategies + non_cooperative_strategies)
                print("cooperative ratio: ", cooperative_strategies / (cooperative_strategies + non_cooperative_strategies))
                print("non-cooperative ratio: ", non_cooperative_strategies / (cooperative_strategies + non_cooperative_strategies))
                '''

                cooperative_strategies = []
                non_cooperative_strategies = []
                for agent in agents:
                    if agent.strategy <= 0:
                        cooperative_strategies.append(1)
                    else:
                        non_cooperative_strategies.append(1)

                print("number of cooperative agents: ", len(cooperative_strategies))
                print("number of non-cooperative agents: ", len(non_cooperative_strategies))

                print("sum: ", len(cooperative_strategies) + len(non_cooperative_strategies))
                print("cooperative ratio: ",
                      len(cooperative_strategies) / (len(cooperative_strategies) + len(non_cooperative_strategies)))
                print("non-cooperative ratio: ",
                      len(non_cooperative_strategies) / (len(cooperative_strategies) + len(non_cooperative_strategies)))

                print("average cooperation ratio: ", (cooperative_interactions / all_interactions))

                plt.legend()
                plt.savefig('strategies-at-{}.pdf'.format(generation))
                plt.show()

        for agent in agents:
            last_generations.append(agent.strategy)

    return lines_graph, coalitions_size, cooperation_ratio, last_generations


def agent_wants_best_neighbour_strategy_without_old_values(agent, network):
    neighbours = []
    neighbours_old_values = []

    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):
                neighbours.append(network.nodes[neighbor]['agent'])
                neighbours_old_values.append(network.nodes[neighbor]['agent'].strategy)

    neighbours_payoffs = [neighbour.payoff for neighbour in neighbours]
    max_payoff = max(neighbours_payoffs)

    for i in range(len(neighbours)):
        if neighbours[i].payoff == max_payoff:
            if max_payoff > agent.payoff:
                return neighbours_old_values[i]
            else:
                return agent.strategy


def agent_wants_best_neighbour_strategy(agent, network):
    neighbours = []
    neighbours_old_values = []

    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):
                neighbours.append(network.nodes[neighbor]['agent'])
                neighbours_old_values.append(network.nodes[neighbor]['oldValue'])
                # neighbours_old_values.append(network.nodes[neighbor]['agent'].strategy)

    neighbours_payoffs = [neighbour.payoff for neighbour in neighbours]
    max_payoff = max(neighbours_payoffs)

    for i in range(len(neighbours)):
        if neighbours[i].payoff == max_payoff:
            if max_payoff > agent.payoff:
                return neighbours_old_values[i]
            else:
                return agent.strategy
    '''
    for neighbour in neighbours:
        if neighbour.payoff == max_payoff:
            return neighbour.strategy
    '''


def agent_wants_best_coalition(agent, network, coalitions):
    coalitions_to_use = []
    coalitions_score = []

    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):

                neighbor_coalition = get_agent_coalition(network.nodes[neighbor]['agent'], coalitions)
                if neighbor_coalition is not None:
                    coalitions_to_use.append(neighbor_coalition)
                    coalitions_score.append(coalition_score(neighbor_coalition))

    agent_coalition = get_agent_coalition(agent, coalitions)
    if agent_coalition is not None:
        coalitions_to_use.append(agent_coalition)
        coalitions_score.append(coalition_score(agent_coalition))

    if len(coalitions_score) == 0:
        return None, None
    else:
        max_coalition_score = coalitions_score[0]
        best_coalition_to_use = coalitions_to_use[0]

        for i in range(len(coalitions_score)):
            if coalitions_score[i] > max_coalition_score:
                max_coalition_score = coalitions_score[i]
                best_coalition_to_use = coalitions_to_use[i]

        return best_coalition_to_use, max_coalition_score


def agent_wants_best_independent_neighbor(agent, network, coalitions):
    independent_neighbours = []

    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):

                agent_coalition = [coalition for coalition in coalitions if
                                   network.nodes[neighbor]['agent'] in coalition]
                if len(agent_coalition) == 0:
                    independent_neighbours.append(network.nodes[neighbor]['agent'])

    if len(independent_neighbours) == 0:
        return None, None
    else:
        best_score = independent_neighbours[0].image_score
        best_neighbour = independent_neighbours[0]

        for neighbour in independent_neighbours:
            if neighbour.image_score > best_score:
                best_score = neighbour.image_score
                best_neighbour = neighbour

        return best_neighbour, best_score


def agent_was_worst_in_neighbors(agent, network):
    neighbors_payoffs = []

    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):
                neighbors_payoffs.append(network.nodes[neighbor]['agent'].payoff)

    min_payoff = min(neighbors_payoffs)

    if agent.payoff <= min_payoff:
        return True
    else:
        return False


def anyNeighbor_in_agents_coalition(agent, coalitions, network):
    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == agent:
            for neighbor in network.neighbors(node):
                for coalition in coalitions:
                    if network.nodes[neighbor]['agent'] in coalition:
                        return True

            return False


def coalition_score(coalition):
    return sum([agent.image_score for agent in coalition]) / float(len(coalition))


def is_peer(donor, recipient, coalitions):
    for coalition in coalitions:
        if donor in coalition and recipient in coalition:
            return True
    return False


def is_neighbor(donor, recipient, network):
    for node, node_data in network.nodes(data=True):
        if network.nodes[node]['agent'] == donor:
            for neighbor in network.neighbors(node):
                if network.nodes[neighbor]['agent'] == recipient:
                    return True
    return False


def get_agent_coalition(agent, coalitions):
    for coalition in coalitions:
        if agent in coalition:
            return coalition
    return None


def find_players(agents):
    a_i, a_j = random.sample(agents, 2)
    while a_i == a_j:
        a_i, a_j = random.sample(agents, 2)
    return a_i, a_j


cooperations = []
last_generation_average = []
for i in range(100):
    print("Estamos no ", i)
    a, coalition_size, cooperation, last_generation = play(to_draw=False, barabasi=False)
    # Let's save all the strategys
    last_generation_average.append(sum(last_generation)/len(last_generation))

plt.hist(last_generation_average, density=True)
plt.savefig('last_generation_average_strogatz.pdf')
plt.show()
'''
a, barabasi_coalition_size, barabasi_cooperation = play(to_draw=True, barabasi=True)
d, strogatz_coalition_size, strogatz_cooperation = play(to_draw=True, barabasi=False)

plt.plot(barabasi_cooperation, label="Scale-free", color="#2F80ED")
plt.plot(strogatz_cooperation, '--', label="Small-world", color="#27AE60")

plt.savefig('cooperation_stuff.pdf')
plt.show()

plt.plot(barabasi_coalition_size, label="Scale-free", color="#2F80ED")
plt.plot(strogatz_coalition_size, '--', label="Small-world", color="#27AE60")
plt.savefig('coalition_stuff.pdf')
plt.show()
'''


