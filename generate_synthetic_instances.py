import numpy as np

# %% Define budget fractions

budget_fractions = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]

# %%


def write_file(nodes, edges, weights, budgets, folder_name, file_name, weight_type='int'):

    # Open file
    f = open('collections/{:s}/{:s}'.format(folder_name, file_name), 'w')

    # Write header
    n_nodes = len(nodes)
    n_edges = len(edges)
    f.write('{:d} {:d} {:s}\n'.format(n_nodes, n_edges, weight_type))

    # Write edges
    for (i, j) in edges:
        if weight_type == 'int':
            f.write('{:d} {:d} {:d}\n'.format(i, j, edges[(i, j)]))
        else:
            f.write('{:d} {:d} {:.6f}\n'.format(i, j, edges[(i, j)]))

    # Write weights
    for weight in weights:
        f.write('{:d} '.format(weight))
    f.write('\n')

    # Write budgets
    for budget in budgets:
        f.write('{:d} '.format(budget))

    f.close()


def generate_geometrical_problem_instance(n_nodes):

    # Generate random locations within a 100x100 square for n nodes
    locations = np.random.rand(n_nodes, 2) * 100

    # Calculate the Euclidean distance matrix between the points
    distances = np.sqrt(np.sum((locations[:, np.newaxis] - locations[np.newaxis, :]) ** 2, axis=2))

    return distances


def generate_weighted_geometrical_problem_instance(n_nodes):

    # Generate random locations within a 100x100 square for n nodes
    locations = np.random.rand(n_nodes, 2) * 100

    # Assign random weights to each location within the range [5, 10]
    weights = np.random.uniform(5, 10, size=n_nodes)

    # Calculate the weighted Euclidean distance matrix between the points
    distances = np.outer(weights, weights) * np.sqrt(
        np.sum((locations[:, np.newaxis] - locations[np.newaxis, :]) ** 2, axis=2))

    return distances


def generate_exponential_problem_instance(n_nodes, mean=50):

    # Generate a distance matrix where each entry is drawn from an exponential distribution
    distances = np.random.exponential(mean, size=(n_nodes, n_nodes))

    # Make the distance matrix symmetric since distance from i to j is the same as from j to i
    distances = (distances + distances.T) / 2

    # Set the diagonal to 0 as the distance from a point to itself is 0
    np.fill_diagonal(distances, 0)

    return distances


def generate_random_problem(n_nodes):

    # Generate a distance matrix where each entry is a random integer between 1 and 100
    distances = np.random.randint(1, 101, size=(n_nodes, n_nodes))

    # Make the distance matrix symmetric
    distances = (distances + distances.T) // 2

    # Set the diagonal to 0 as the distance from a point to itself is 0
    np.fill_diagonal(distances, 0)

    return distances


def generate_weights(n_nodes):

    # Generate random weights for each location, each an integer between 1 and 100
    weights = np.random.randint(1, 101, size=n_nodes)

    return weights


# %% Generate instances of Dispersion-QKP collection

# Number of nodes
n_nodes_values = [300, 500, 1000, 2000]
instance_types = ['geo', 'wgeo', 'expo', 'ran']
sparsification_fractions = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

# Generate instances for each value of n_nodes_values
for n_nodes in n_nodes_values:

    # Set random seed
    np.random.seed(24)

    # Generate nodes
    nodes = np.arange(n_nodes)

    # Get weights
    weights = generate_weights(n_nodes)

    for instance_type in instance_types:

        # Sparsify utility matrix
        for sparsification_fraction in sparsification_fractions:

            # Generate edges
            if instance_type == 'geo':
                utility_matrix = generate_geometrical_problem_instance(n_nodes)
            elif instance_type == 'wgeo':
                utility_matrix = generate_weighted_geometrical_problem_instance(n_nodes)
            elif instance_type == 'expo':
                utility_matrix = generate_exponential_problem_instance(n_nodes)
            else:
                utility_matrix = generate_random_problem(n_nodes)

            # Sparsify utility matrix
            utility_matrix = utility_matrix * (np.random.rand(n_nodes, n_nodes) < sparsification_fraction)

            edges = {}
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if utility_matrix[i, j] > 0:
                        edges[i, j] = utility_matrix[i, j]

            # Add budgets
            budgets = []
            for budget_fraction in budget_fractions:
                budgets.append(int(budget_fraction * np.sum(weights)))

            write_file(nodes, edges, weights, budgets, 'Dispersion-QKP',
                       'dispersion-qkp-{:s}_{:d}_{:d}.txt'.format(instance_type, n_nodes,
                                                                  int(sparsification_fraction * 100)),
                       weight_type='float')


# %% Generate instances of TeamFormation-QKP-2 collection

n_participants_values = [1000, 2000, 4000, 6000, 8000, 10000]
n_projects = 30000  # Must be larger than max number of projects to choose for a participant
# (depends on lognormal params)
log_normal_mean = 4
log_normal_standard_deviation = 1
max_weight = 10

for n_participants in n_participants_values:

    # Set random seed
    np.random.seed(24)

    # Generate set of projects
    projects = np.arange(n_projects)

    # Step 1: generate subsets
    indices = []
    counter = 0
    while counter < n_projects:
        cardinality = 1 + int(np.random.lognormal(log_normal_mean, log_normal_standard_deviation))
        counter += cardinality
        indices.append(counter)
    subsets = np.split(projects, indices)[:-1]
    n_subsets = len(subsets)

    # Step 2: determine number of projects per participant
    n_projects_per_participant = []
    for i in range(n_participants):
        cardinality = 1 + int(np.random.lognormal(log_normal_mean, log_normal_standard_deviation))
        n_projects_per_participant.append(cardinality)

    # Step 3: choose projects of each participant
    projects_dict = dict()
    for i in range(n_participants):
        subset_id = np.random.randint(n_subsets)
        cardinality_of_subset = len(subsets[subset_id])
        if n_projects_per_participant[i] < cardinality_of_subset:
            selected_projects = np.random.choice(subsets[subset_id], n_projects_per_participant[i], replace=False)
        else:
            selected_projects = subsets[subset_id]
            remaining_projects = np.setdiff1d(projects, selected_projects)
            n_projects_to_choose = n_projects_per_participant[i] - cardinality_of_subset
            selected_projects = np.concatenate((selected_projects, np.random.choice(remaining_projects,
                                                                                    n_projects_to_choose,
                                                                                    replace=False)))
        projects_dict[i] = selected_projects

    # Step 4: compute Jaccard similarity
    utility_matrix = np.zeros((n_participants, n_participants))
    for i in range(n_participants):
        for j in range(i+1, n_participants):
            n_joint_projects = len(np.intersect1d(projects_dict[i], projects_dict[j]))
            n_projects_i = len(projects_dict[i])
            n_projects_j = len(projects_dict[j])
            utility_matrix[i, j] = n_joint_projects/(n_projects_i + n_projects_j - n_joint_projects)
            utility_matrix[j, i] = n_joint_projects / (n_projects_i + n_projects_j - n_joint_projects)

    # Display density of utility matrix
    density = np.count_nonzero(utility_matrix) / utility_matrix.size
    print('Density of utility matrix: {:.4f}'.format(density))

    edges = {}
    for i in range(n_participants):
        for j in range(i + 1, n_participants):
            if utility_matrix[i, j] > 0:
                edges[i, j] = utility_matrix[i, j]

    # Generate nodes
    nodes = np.arange(n_participants)

    # Simulate weights
    weights = np.random.randint(1, max_weight + 1, n_participants)

    # Add budgets
    budgets = []
    for budget_fraction in budget_fractions:
        budgets.append(int(budget_fraction * np.sum(weights)))

    write_file(nodes, edges, weights, budgets, 'TeamFormation-QKP-2',
               'synthetic_tf_{:d}.txt'.format(n_participants), weight_type='float')

# %% Generate instances of New-QKP collection

# Read raw_data
instances_folder = 'New-QKP'
n_nodes_list = [500, 1000, 2000, 5000, 10000]
densities = {500: [5, 10, 15, 20, 25, 50, 75, 100],
             1000: [5, 10, 15, 20, 25, 50],
             2000: [5, 10, 15, 20, 25],
             5000: [5, 10, 15, 20],
             10000: [5]}
n_seeds = 1

# Set random seed
np.random.seed(24)

for n_nodes in n_nodes_list:

    # Get nodes
    nodes = np.arange(n_nodes)

    for density in densities[n_nodes]:
        for seed in range(n_seeds):

            # Randomly draw linear and quadratic coefficients in [1, 100]
            utility_matrix = np.random.randint(1, 101, size=(n_nodes, n_nodes))
            utility_matrix = np.tril(utility_matrix) + np.tril(utility_matrix, -1).T

            # Apply density
            utility_matrix = utility_matrix * (np.random.rand(n_nodes, n_nodes) < (density / 100))

            edges = {}
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if utility_matrix[i, j] > 0:
                        edges[i, j] = utility_matrix[i, j]

            # Randomly draw weights in [1, 50]
            weights = np.random.randint(1, 51, size=n_nodes)

            # Add budgets
            budgets = []
            for budget_fraction in budget_fractions:
                budgets.append(int(budget_fraction * np.sum(weights)))

            write_file(nodes, edges, weights, budgets, 'New-QKP',
                       'qkp_new_{:d}_{:d}_{:d}.txt'.format(n_nodes, density, seed), weight_type='float')
