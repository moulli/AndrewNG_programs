import numpy as np
import random
import copy

# Hippolyte Moulle - 30/09/2022
# A series of notes:
# FITNESS SHOULD BE AN ARRAY OF POSITIVE VALUES, otherwise this will not work.
# The code is not optimized, there should be a way to make it faster.
# At the moment, individuals resulting from crossovers are not mutated, I do not know if it is right.
# I am not sure about the speciation process. I favoured the current species for each individual, so that some species do not vanish just after they come, due to the randomness of the selected representative.
# When a species fitness does not improve for a certain number of generations, its adjusted fitness is set to 0. I don't know if that is the case in the paper, it only says the species is not allowed to reproduce.
# I did not implement the specificity that if the fitness of the entire population does not improve for more than 20 generations, only the top two species are allowed to reproduce.





# NETWORK CLASS
class network:
	
	# Initialize
	def __init__(self, n_inputs, n_outputs, shifts):
		"""Define network.
		shift is a list with mean and std for normal distribution sampling leading to weights."""
		
		# Save number of inputs, number of outputs and number of hidden neurons (0 in the beginning)
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_hiddens = 0
		
		# Define genes
		self.genes = []
		for no in range(n_outputs):
			for ni in range(1+n_inputs):
				weight = np.random.normal(loc=shifts[0], scale=shifts[1])
				self.genes.append([ni, 1+n_inputs+no, weight, True, (1+n_inputs)*no+ni])

		# Define output transfer function
		self.output_transfer = self.sigmoid
				
		# Define current state and transition matrix
		self.reinit()
		
	# Define current state function
	def compute_current_state(self):
		"""Computes current state, simply resets the current state to a vector of zeros."""
		
		# Reinitialize current state
		self.current_state = np.zeros((self.n_outputs+self.n_hiddens,))        
				
	# Define transition matrix function
	def compute_T(self):
		"""Computes transition matrix."""
		
		# We are going to save the hidden nodes with a different index, it is going to be easier to manipulate
		dict_hiddens = {}
		all_nodes = list(set([gene[0] for gene in self.genes] + [gene[1] for gene in self.genes]))
		for an in all_nodes:
			if an < 1+self.n_inputs+self.n_outputs:
				dict_hiddens[an] = an
			else:
				dict_hiddens[an] = max([val for val in dict_hiddens.values()]) + 1
		# Build transition matrix
		self.T = np.zeros((self.n_outputs+self.n_hiddens, 1+self.n_inputs+self.n_outputs+self.n_hiddens))
		for gene in self.genes:
			if gene[3]:                    
				self.T[dict_hiddens[gene[1]]-1-self.n_inputs, dict_hiddens[gene[0]]] = gene[2]
				
	# Reinitialize
	def reinit(self):
		"""Compute/reinitialize current state, and compute transition matrix."""
		
		# Compute current state
		self.compute_current_state()
		
		# Compute transition matrix
		self.compute_T()
		
	# Transfer functions
	def sigmoid(self, x):
		"""Sigmoid function used in NEAT paper."""
		
		return 1/(1+np.exp(-4.9*x))
	def tanh(self, x):
		"""Simple hyperbolic tangent function."""

		return np.tanh(x)
		
	# Update
	def update(self, inputs):
		"""Computes the new current state, based on former state, and inputs."""
		
		# Add bias to inputs, to current state
		current_state = np.hstack((np.ones((1,)), np.squeeze(inputs), self.current_state))
		
		# Compute next state
		next_state = np.dot(self.T, current_state)
		#Do transfer functions
		next_state[:self.n_outputs] = self.output_transfer(next_state[:self.n_outputs])
		next_state[self.n_outputs+1:] = self.sigmoid(next_state[self.n_outputs+1:])
		
		# Save new state as current state
		self.current_state = next_state
		
		# Return outputs
		return next_state[:self.n_outputs]
	
	# Mutate weights
	def mutate_weights(self, prob_mutation, prob_uniform, shifts):
		"""Mutates weights with a probability of prob_mutation.
		Mutates uniformly with a probability of prob_uniform, or assigns a new value."""
		
		# Mutate weights ?
		if random.random() < prob_mutation:
			# Sample from uniform distribution to add to weight in case of uniform mutation
			w_uniform = np.random.normal(loc=shifts[0], scale=shifts[1])
			
			# Loop over each weight
			for gi in range(len(self.genes)):
				# Get weight
				old_weight = self.genes[gi][2]
				# Uniform mutation ?
				if random.random() < prob_uniform:
					new_weight = old_weight + w_uniform
				else:
					new_weight = np.random.normal(loc=shifts[0], scale=shifts[1])
				# Replace weight
				self.genes[gi][2] = new_weight
				
			# Compute current state and transition matrix
			self.reinit()
				
	# Add node
	def mutate_node(self, prob_mutation, shifts, dict_nodes, dict_innov):
		"""Add node with a probability of prob_mutation.
		Information on global NEAT evolution are taken from dict_nodes and dict_innov, and these are updated."""
		
		# Add node ?
		if random.random() < prob_mutation:
			
			# All current nodes
			current_nodes = list(set([gene[0] for gene in self.genes] + [gene[1] for gene in self.genes]))
			# Make a list of the new possible nodes
			possible_nodes_all = [[gene[0], gene[1], gi] for gi, gene in enumerate(self.genes) if gene[0]!=gene[1]] # add index to find it later
			# Check if network already has some of these nodes
			possible_nodes = []
			for pna in possible_nodes_all:
				if tuple(pna[:-1]) in dict_nodes.keys() and dict_nodes[tuple(pna[:-1])] in current_nodes:
					continue
				else:
					possible_nodes.append(pna)
			# Select a new node
			new_node = random.choice(possible_nodes)
			
			# Does it need a new number ?
			if tuple(new_node[:-1]) in dict_nodes.keys():
				node_num = dict_nodes[tuple(new_node[:-1])]
			else:
				node_num = max(dict_nodes["bias"] + dict_nodes["inputs"] + dict_nodes["outputs"] + [nn for nn in dict_nodes.values() if not isinstance(nn, list)]) + 1
				# Update nodes dictionnary
				dict_nodes[tuple(new_node[:-1])] = node_num
				dict_nodes[tuple(new_node[:-1][::-1])] = node_num
				
			# What are the innovation numbers ?
			if (new_node[0], node_num) in dict_innov.keys():
				innov_in_num = dict_innov[(new_node[0], node_num)]
			else: 
				innov_in_num = max([ni for ni in dict_innov.values()]) + 1
				# Update innovation dictionnary
				dict_innov[(new_node[0], node_num)] = innov_in_num
			if (node_num, new_node[1]) in dict_innov.keys():
				innov_out_num = dict_innov[(node_num, new_node[1])]
			else: 
				innov_out_num = max([ni for ni in dict_innov.values()]) + 1
				# Update innovation dictionnary
				dict_innov[(node_num, new_node[1])] = innov_out_num
				
			# Disable old connexion
			self.genes[new_node[2]][3] = False
			# Add in connexion
			self.genes.append([new_node[0], node_num, 1, True, innov_in_num])
			self.genes.append([node_num, new_node[1], random.uniform(shifts[0], shifts[1]), True, innov_out_num])
			
			# Add hidden neuron to record
			self.n_hiddens += 1
			# Compute current state and transition matrix
			self.reinit()
				
		return dict_nodes, dict_innov
	
	# Add connexion
	def mutate_connexion(self, prob_mutation, shifts, dict_innov):
		"""Add connexion with a probability of prob_mutation.
		Information on global NEAT evolution are taken from dict_innov, and it is updated."""
				
		# Add connexion ?
		if random.random() < prob_mutation: 
			
			# We are first going to rebuild the hidden node dictionnary in reverse, to go from T to the associate nodes
			dict_hiddens = {}
			all_nodes = list(set([gene[0] for gene in self.genes] + [gene[1] for gene in self.genes]))
			for an in all_nodes:
				if an < 1+self.n_inputs+self.n_outputs:
					dict_hiddens[an] = an
				else:
					dict_hiddens[max([key for key in dict_hiddens.keys()]) + 1] = an
			
			# All possible connexions
			possible_connexions = []
			for ti in range(self.T.shape[0]):
				for tj in range(self.T.shape[1]):
					if self.T[ti, tj] == 0:
						possible_connexions.append([dict_hiddens[tj], dict_hiddens[ti+1+self.n_inputs]])
			
			# If there is in deed a connexion available
			if 0 < len(possible_connexions):
				# Select a new connexion
				new_connexion = random.choice(possible_connexions)
				# Does it need a new innovation number ?
				if tuple(new_connexion) in dict_innov.keys():
					innov_num = dict_innov[tuple(new_connexion)]
				else:
					innov_num = max([ni for ni in dict_innov.values()]) + 1
					# Update innovation dictionnary
					dict_innov[tuple(new_connexion)] = innov_num

				# Add new connexion as a gene
				self.genes.append([new_connexion[0], new_connexion[1], random.uniform(shifts[0], shifts[1]), True, innov_num])
				# Compute current state and transition matrix
				self.reinit()
				
		return dict_innov





class NEAT:
	
	# Initialize
	def __init__(self, num_nets, n_inputs, n_outputs, shifts):
		"""We are going to create num_nets instance of the network class."""
		
		# Save parameters
		self.num_nets = num_nets
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.shifts = shifts
		
		# Create networks
		self.networks = []
		for n in range(num_nets):
			self.networks.append(network(n_inputs=n_inputs, n_outputs=n_outputs, shifts=shifts))
			
		# Record nodes and innovation number
		self.nodes = {"bias": [0], "inputs": [ni+1 for ni in range(n_inputs)], "outputs": [no+1+n_inputs for no in range(n_outputs)]}
		self.innov = {}   
		for no in range(n_outputs):
			for ni in range(1+n_inputs):
				self.innov[(ni, 1+n_inputs+no)] = (1+n_inputs)*no+ni
			
		# Define species
		self.species = np.zeros((num_nets,)).astype(int)
		
		# Species representative for speciation
		self.representatives = {0: random.choice(self.networks)}
		
		# Maximum fitness per species
		self.max_fit = {0: [0, 0]}
		
	# Compute compatibility distance
	def compute_compatibility(self, net_i, net_j, c1, c2, c3, N_small):
		"""Computes compatibility distance between networks net_i and net_j.
		If the compatibility distance is below a certain threshold, that means net_i and net_j are part of the same species."""
		
		# Innovation number of both networks
		innov = [[gene[4] for gene in net_i.genes], [gene[4] for gene in net_j.genes]]
		
		# Number of excess genes
		excess_network = np.argmax([np.max(innov[0]), np.max(innov[1])])
		E = np.sum(np.max(innov[1-excess_network]) < innov[excess_network])
		
		# Number of disjoint genes
		D = 0
		for ii in innov[1-excess_network]:
			if ii not in innov[excess_network]:
				D += 1
		for ii in innov[excess_network]:
			if ii not in innov[1-excess_network] and ii <= np.max(innov[1-excess_network]):
				D += 1
				
		# Average weight difference of matching genes
		W = np.zeros((0,))
		for ii in innov[1-excess_network]:
			if ii in innov[excess_network]:
				ind_i = innov[0].index(ii)
				ind_j = innov[1].index(ii)
				W = np.hstack((W, net_i.genes[ind_i][2]-net_j.genes[ind_j][2]))
		W_ = np.abs(np.mean(W))
		
		# Number of genes in the larger genome
		N = np.max([len(net_i.genes), len(net_j.genes)])
		if N < N_small: # if not many genes, set N to 1
			N = 1
			
		# Compute compatibility distance
		delta = c1*E/N + c2*D/N + c3*W_
	
		return delta
			
	# Speciate
	def speciate(self, c1, c2, c3, deltat, N_small):
		"""This function speciates the new networks.
		c1, c2, c3 are coefficients used to compute compatibility distances.
		In order to speciate, if N < N_small, then N is set to 1. This (probably) allows easier speciation in the beginning.
		deltat is the compatibility distance to determine species belonging.
		%%%% IMPORTANT! READ ME :
		%%%% There is an interesting pattern to be mentionned: as the choice of a representative is random from the previous generation, the order of species you chose to speciate is VERY important.
		%%%% You might have a new species "1", but the next time you speciate, the representative from species "0" looks a lot like an individual from the new species, and it gets assigned to species "1".
		%%%% I do not know how to solve this problem yet, so I am going to allow each species to check if it is compatible to the representative of its own species, before testing the other species.
		%%%% This should add a better stability in the different species."""
		
		# Represented species from former generation
		old_species = np.unique(self.species)
		
		# Loop over networks to see which species it belongs to
		current_species = -np.ones((self.species.shape))
		for n in range(self.num_nets):
			
			# Network
			net_i = self.networks[n]
			# Loop over existing species representatives to compute compatibility distance
			for rep in np.hstack((self.species[n], old_species)): # preference for current species before looping on other species
				# Get representative
				net_j = self.representatives[rep]
				# Compute compatibility distance
				delta = self.compute_compatibility(net_i, net_j, c1, c2, c3, N_small)
				# Compare it to threshold distance
				if delta <= deltat:
					# Belongs to same species
					current_species[n] = rep
					break
					
			# If no species was found, create a new one
			if current_species[n] == -1:
				# Add new species number
				new_spec_num = np.max([key for key in self.max_fit.keys()])+1
				current_species[n] = new_spec_num
				# Add to old species to be able to assign other individuals
				old_species = np.hstack((old_species, new_spec_num.astype(int)))
				# Add representative
				self.representatives[new_spec_num] = self.networks[n]
				# Add max fit to be able to remove it later
				self.max_fit[new_spec_num] = [0, 0]
		
		# Save new species array
		self.species = current_species.astype(int)
		
		# Find representatives from species
		self.find_representatives()
		
	# Find representative in each species
	def find_representatives(self):
		"""Loop over species to select representative networks."""
		
		# Represented species
		all_species = np.unique(self.species)
		
		# Find representative
		for spec in all_species:
			# All individuals in that species
			indivs = np.where(self.species==spec)[0]
			# Take random one
			representative = np.random.choice(indivs)
			# Add representative to dictionnary
			self.representatives[spec] = self.networks[representative]
			
	# Compute adjusted fitness
	def compute_adjusted_fitness(self, fitness, max_shit_gen):
		"""Computes adjusted fitness based on species.
		max_shit_gen is the number of generations after which a species can not reproduce if its fitness has not improved."""
		
		# Recompute fitness
		adjusted_fitness = np.zeros((fitness.shape))
		for f in range(len(fitness)):
			adjusted_fitness[f] = fitness[f] / np.sum(self.species==self.species[f])
			
		# All species
		all_species = np.unique(self.species)
			
		# Update species max fitness, and remove species which have been lame for a while
		for spec in all_species:
			# Maximum fitness
			max_fit_spec = np.max(fitness[self.species==spec])
			# Compare to old best fitness
			if self.max_fit[spec][0] < max_fit_spec:
				self.max_fit[spec][0] = max_fit_spec
				self.max_fit[spec][1] = 0
			else:
				self.max_fit[spec][1] += 1
			# Remove lame species
			if max_shit_gen <= self.max_fit[spec][1]:
				adjusted_fitness[self.species==spec] = 0
				
		return adjusted_fitness
		
	# Selection process
	def select(self, fitness, max_shit_gen, champ, perc_rep):
		"""This function updates the fitness using speciation, and selects which networks will be able to reproduce.
		fitness is the reward of the reinforcement learning task.
		max_shit_gen is the number of generations after which a species can not reproduce if its fitness has not improved.
		champ is the size the population must be to have its champion replicated in the next generation.
		perc_rep is the proportion of networks allowed to reproduce."""
					
		# Compute adjusted fitness
		adjusted_fitness = self.compute_adjusted_fitness(fitness, max_shit_gen)
		
		# All species
		all_species = np.unique(self.species)
			
		# For each species, compute number of offsprings
		num_offsprings = np.zeros((np.max(all_species)+1,))
		for no in range(len(num_offsprings)):
			num_offsprings[no] = np.round(np.sum(adjusted_fitness[self.species==no]) / np.sum(adjusted_fitness) * self.num_nets)
			
		# If total number of offsprings differs from population number, adjust it
		tot_offsprings = np.sum(num_offsprings)
		if tot_offsprings < self.num_nets:
			for dif in range(int(self.num_nets-tot_offsprings)):
				spec = np.random.choice(all_species)
				num_offsprings[spec] += 1
		elif self.num_nets < tot_offsprings:
			for dif in range(int(tot_offsprings-self.num_nets)):
				# Find species with more than 0 individuals
				spec_ok = np.where(0 < num_offsprings)[0]
				spec = np.random.choice(spec_ok)
				num_offsprings[spec] -= 1
				
		# Select champions to be copied to next generation
		champions = {}
		for spec in all_species:
			# Recover individuals from species
			indivs = np.where(self.species==spec)[0]
			# If number of individuals is higher than champ, save champion
			if champ <= len(indivs):
				max_fit_ind = np.argmax(adjusted_fitness[indivs])
				champions[spec] = indivs[max_fit_ind]
		
		# Select networks able to reproduce
		ok_to_reproduce = {}
		for spec in all_species:
			# Recover individuals from species
			indivs = np.where(self.species==spec)[0]
			# Recover perc_rep % of individuals with highest fitness
			sorted_fit = np.argsort(-adjusted_fitness[indivs])
			sorted_fit_values = np.sort(-adjusted_fitness[indivs])
			num_keep = np.ceil(perc_rep * len(sorted_fit))
			# Fill dictionnary
			ok_to_reproduce[spec] = []
			for kip in range(int(num_keep)):
				ok_to_reproduce[spec].append((indivs[sorted_fit[kip]], -sorted_fit_values[kip]))
			
		return num_offsprings, champions, ok_to_reproduce
		
	# Offspring generation
	def make_offspring(self, net_i, net_j, prob_disabled):
		"""From two networks, creates an offspring.
		net_i and net_j are the two network, net_i being the one with the highest fitness. 
		prob_disabled is the probability a gene is disabled if it is disabled in either parent."""
		
		# Create new network (with the basic inputs/outputs)
		offspring = network(n_inputs=self.n_inputs, n_outputs=self.n_outputs, shifts=self.shifts)
		# Node list
		offspring_nodes = list(set([gene[0] for gene in offspring.genes] + [gene[1] for gene in offspring.genes]))
		
		# List innovation numbers
		innovs = [[gene[4] for gene in net_i.genes], [gene[4] for gene in net_j.genes], [gene[4] for gene in offspring.genes]]
		
		# Modify offsping genes by looping on more fit parent genes
		for ii, innov in enumerate(innovs[0]):
			# Check if gene is in other parent
			if innov in innovs[1]:
				# Take randomly one or the other
				new_gene = random.choice([net_i.genes[ii], net_j.genes[innovs[1].index(innov)]])
				# If it is deactivated in either parent, deactivate it for new gene (might be reactivated later)
				new_gene[3] = np.all([net_i.genes[ii][3], net_j.genes[innovs[1].index(innov)][3]])
			else:
				new_gene = net_i.genes[ii]
			# Potential reactivation of gene
			if not new_gene[3] and prob_disabled < random.random():
				new_gene[3] = True
			# Replace or add gene to offspring
			if innov in innovs[2]:
				offspring.genes[innovs[2].index(innov)] = new_gene
			else:
				offspring.genes.append(new_gene)
				
		# Increase number of hidden nodes records in offspring if needed
		new_nodes = list(set([gene[0] for gene in offspring.genes] + [gene[1] for gene in offspring.genes]))
		for nn in new_nodes:
			if nn not in offspring_nodes:
				offspring.n_hiddens += 1
		# Reinit offspring
		offspring.reinit()
			
		return offspring
	
	# Next generation creation
	def generate(self, num_offsprings, champions, ok_to_reproduce, p_no_cross, p_interspecies, p_disabled, p_weights, p_uniform, p_node, p_connexion, p_connexion_larger_pop):
		"""Generates next generation after selection, using mating and mutations.
		num_offsprings is the number of offsprings allowed per species.
		champions includes each champion for species with more than a certain number of individuals.
		ok_to_reproduce includes the top individuals in each species, able to reproduce.
		p_no_cross represent the proportion of individuals that will only undergo mutations, no mating.
		p_interprecies in the probability an individual mates with another individual from another species.
		p_disabled is used in crossing,
		p_weight, p_uniform, p_node, p_connexion and p_connexion_larger_pop are used in mutations.
		p_connexion_larger_pop is p_connexion if individual is from the larger population."""
		
		# Define outputs
		new_species = np.zeros((0,))
		new_networks = []
		
		# We do not want to mutate the champions
		mutate_these = np.zeros((0,))
		
		# Loop over all species
		all_species = np.unique(self.species)
		for spec in all_species:
			# Add as many new individuals as the species can
			for nof in range(int(num_offsprings[spec])):
				# Save individual species
				new_species = np.hstack((new_species, spec))
				
				# Save champion if it exists, otherwise get to the rest
				if nof == 0 and spec in champions.keys(): # add champion
					# Do not mutate champion
					mutate_these = np.hstack((mutate_these, 0))
					# Add champion
					network_ = copy.deepcopy(self.networks[champions[spec]])
					new_networks.append(network_)
					
				else:
					# These will be mutated
					#mutate_these = np.hstack((mutate_these, 1))
					
					# Only mutate or make offspring
					if nof / num_offsprings[spec] <= p_no_cross:
						mutate_these = np.hstack((mutate_these, 1)) # TO MAYBE REMOVE LATER
						network_ = copy.deepcopy(self.networks[random.choice(ok_to_reproduce[spec])[0]])
						new_networks.append(network_)
						
					else:
						mutate_these = np.hstack((mutate_these, 0)) # TO MAYBE REMOVE LATER
						# First individual
						net_i_choice = random.choice(ok_to_reproduce[spec])
						net_i = copy.deepcopy(self.networks[net_i_choice[0]])
						net_i_fitness = net_i_choice[1]
						# Second individual, through intra or inter species crossing
						if p_interspecies < random.random():
							net_j_choice = random.choice(ok_to_reproduce[spec])
							net_j = copy.deepcopy(self.networks[net_j_choice[0]])
							net_j_fitness = net_j_choice[1]
						else:
							# Find second network
							other_spec_all = []
							for other_spec, num_off in enumerate(num_offsprings):
								if 0 < num_off and other_spec != spec:
									other_spec_all.append(other_spec)
							# If no other species, take from same species
							if len(other_spec_all) == 0:
								net_j_choice = random.choice(ok_to_reproduce[spec])
								net_j = copy.deepcopy(self.networks[net_j_choice[0]])
								net_j_fitness = net_j_choice[1]
							else:
								net_j_choice = random.choice(ok_to_reproduce[random.choice(other_spec_all)])
								net_j = copy.deepcopy(self.networks[net_j_choice[0]])
								net_j_fitness = net_j_choice[1]
						# Create offspring
						if net_i_fitness < net_j_fitness: # switch according to fitness
							net_temp = net_i
							net_i = net_j
							net_j = net_temp
						offspring = self.make_offspring(net_i, net_j, p_disabled)
						# Add offspring to new networks
						new_networks.append(offspring)
					
		# Update species
		self.species = new_species.astype(int)
						
		# Mutate new individuals
		for inet in range(len(new_networks)):
			# Do mutations on appropriate networks
			if mutate_these[inet] == 1:
				# Copy network
				net_i = copy.deepcopy(new_networks[inet])
				# Weight mutation
				net_i.mutate_weights(p_weights, p_uniform, self.shifts)
				# Node mutation
				self.nodes, self.innov = net_i.mutate_node(p_node, self.shifts, self.nodes, self.innov)
				# Connexion mutation
				if self.species[inet] == np.argmax(np.bincount(self.species)): # different mutation probability if individual is in larger population
					p_connexion = p_connexion_larger_pop 
				self.innov = net_i.mutate_connexion(p_connexion, self.shifts, self.innov)
				# Save network
				new_networks[inet] = net_i
			# Reinit networks just in case
			new_networks[inet].reinit()
		
		# Update networks
		self.networks = new_networks





class NEAT_wrapper:
	"""Concatenate all the necessary controls on the NEAT class, with parameters saved inside."""
	
	# Initialize
	def __init__(self, num_nets, n_inputs, n_outputs):
		"""num_nets: number of networks/individuals, n_inputs: number of inputs, n_outputs: number of outputs."""
		
		# Save input parameters
		self.num_nets = num_nets
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		
		# Save global NEAT parameters
		self.mu = 0 # WEIGHTS, mean of normal distribution to sample from
		self.sig = 2.5 # WEIGHTS, std of normal distribution to sample from
		self.shifts = [self.mu, self.sig] # WEIGHTS, parameters for normal distribution to sample from
		self.c1 = 1 # SPECIATION, importance of excess genes
		self.c2 = 1 # SPECIATION, importance of disjoint genes
		self.c3 = 0.4 # SPECIATION, importance of weight difference for matching genes
		self.deltat = 3 # SPECIATION, threshold above which two individuals are considered a different species
		self.N_small = 20 # SPECIATION, allow for easier speciation in the beginning while computing compatibility distances
		self.max_shit_gen = 15 # SELECTION, maximum number of generations after which a species is not allowed to reproduce if its fitness does not improve
		self.champ = 5 # SELECTION, minimum population number to have the individual with the highest fitness copied into next generation
		self.perc_rep = 0.2 # SELECTION, percentage of individuals allowed to mate and mutate to next generation, based on the highest fitnesses
		self.p_no_cross = 0.25 # CROSSING, probability that a next generation network is only mutated, and not crossed from current generation
		self.p_interspecies = 0.001 # CROSSING, probability of interspecies mating
		self.p_disabled = 0.75 # CROSSING, probability that a gene stays disabled in an offspring if it is disabled in either parent
		self.p_weights = 0.8 # MUTATION, probability that the weights are modified
		self.p_uniform = 0.9 # MUTATION, probability that each weight is uniformly modified (i.e. multiplied by a value, in opposition to randomly defined)
		self.p_node = 0.03 # MUTATION, probability of adding a node
		self.p_connexion = 0.05 # MUTATION, probability of adding a connexion (NB: it is 0.05 for "smaller populations" and 0.3 for "larger populations" in the paper)
		self.p_connexion_larger_pop = 0.3 # MUTATION, same as p_connexion, but for larger populations
		
		# Define NEAT
		self.neat = NEAT(num_nets=self.num_nets, n_inputs=self.n_inputs, n_outputs=self.n_outputs, shifts=self.shifts)
		
		# Speciate and find representative
		self.neat.speciate(self.c1, self.c2, self.c3, self.deltat, self.N_small)
		
	# Return species
	def species(self):
		"""Simply returns array of species."""
		
		return self.neat.species
		
	# Network action
	def action(self, net_i, observation):
		"""Update current state of network net_i, and returns the output of it, when the input is observation."""
		
		# Compute 
		output = self.neat.networks[net_i].update(observation)
		
		return output
	
	# Select, cross, and mutate
	def next_gen(self, fitness):
		"""Returns next generation, based on provided fitness."""

		# Select 
		num_offsprings, champions, ok_to_reproduce = self.neat.select(fitness, self.max_shit_gen, self.champ, self.perc_rep)
		
		# Generate
		self.neat.generate(num_offsprings, champions, ok_to_reproduce, self.p_no_cross, self.p_interspecies, self.p_disabled, self.p_weights, self.p_uniform, self.p_node, self.p_connexion, self.p_connexion_larger_pop) 
		
		# Speciate
		self.neat.speciate(self.c1, self.c2, self.c3, self.deltat, self.N_small)
							   
	