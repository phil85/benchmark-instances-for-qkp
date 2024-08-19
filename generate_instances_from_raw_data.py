import os
import zipfile
import pandas as pd
import numpy as np
import urllib.request
import patoolib


def generate_utility_matrix_from_xlsx_file_of_team_formation_dataset(file_name):

    # Set path of Excel files
    path = 'raw_data/real team formation data sets/'

    # Create dataframe from Excel file
    df = pd.read_excel(path + file_name, header=None, index_col=None)

    # Get number of persons
    n_persons = df.shape[0]

    # Get maximum number of collaborators
    max_n_collaborators = df.shape[1]

    # Initialize utility matrix
    utility_matrix = np.zeros((n_persons, n_persons))

    # Get names of nodes
    names = df.iloc[:, 0].tolist()

    # Initialize dictionary to store the number of items per node
    n_projects_dict = dict()
    for i in range(n_persons):
        n_projects_dict[names[i]] = (i, df.iloc[i, 1])

    # Loop over persons
    for i in range(n_persons):

        # Loop over collaborators
        for j in range(2, max_n_collaborators, 2):

            # If there is no collaborator at this position, continue with next person
            if np.isnan(df.iloc[i, j+1]):
                break

            else:
                # Get collaborator
                collaborator = df.iloc[i, j]

                # Get position of collaborator
                collaborator_pos = n_projects_dict[collaborator][0]

                # Get number of items of partner
                n_projects_collaborator = n_projects_dict[collaborator][1]

                # Get number of joint items
                n_joint_projects = df.iloc[i, j+1]

                # Get number of items of node
                n_projects_person = df.iloc[i, 1]

                # Compute Jaccard similarity
                utility_matrix[i, collaborator_pos] = n_joint_projects/(n_projects_person + n_projects_collaborator
                                                                        - n_joint_projects)

    # Set diagonal elements to zero
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    return utility_matrix

# %% Define budget fractions

budget_fractions = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]

# %% Download standard QKP instances

n_nodes_list = [100, 200, 300]
densities = {100: [25, 50, 75, 100],
             200: [25, 50, 75, 100],
             300: [25, 50]}
n_instances = 10

# Download txt file from url
for n_nodes in n_nodes_list:
    for density in densities[n_nodes]:
        for i in range(1, n_instances + 1):
            url = 'https://cedric.cnam.fr/~soutif/QKP/jeu_{:d}_{:d}_{:d}.txt'.format(n_nodes, density, i)

            # Get file name
            file_name = url.split('/')[-1]

            # Download file
            urllib.request.urlretrieve(url, 'raw_data/Standard-QKP/' + file_name)

# %% Download and unzip QKPGroupII instances

# Download zip file from url
folder_name = 'QKPGroupII'
url = 'https://leria-info.univ-angers.fr/%7Ejinkao.hao/QKPDATA/QKPGroupII.zip'
file_name = url.split('/')[-1]
urllib.request.urlretrieve(url, 'raw_data/' + folder_name + '/' + file_name)

# Unzip file
with zipfile.ZipFile('raw_data/' + folder_name + '/' + file_name, 'r') as zip_ref:
    zip_ref.extractall('raw_data')

# Unzip rar file
for i in [25, 50, 75, 100]:
    patoolib.extract_archive('raw_data/' + folder_name + '/1000_{:d}.rar'.format(i), outdir='raw_data/' + folder_name)

    # Copy files from folder to parent folder
    os.system('cp raw_data/' + folder_name + '/1000_{:d}/* raw_data/'.format(i) + folder_name)

    # Remove folder
    os.system('rm -r raw_data/' + folder_name + '/1000_{:d}'.format(i))

    # Delete rar file
    os.system('rm raw_data/' + folder_name + '/1000_{:d}.rar'.format(i))

for i in ['25', '50(1)', '50(2)', '75(1)', '75(2)', '100(1)', '100(2)']:

    patoolib.extract_archive('raw_data/' + folder_name + '/2000_{:s}.rar'.format(i), outdir='raw_data/' + folder_name)

    # Copy files from folder to parent folder
    os.system('cp "raw_data/' + folder_name + '/2000_{:s}/"* raw_data/'.format(i) + folder_name)

    # Remove folder
    os.system('rm -r "raw_data/' + folder_name + '/2000_{:s}"'.format(i))

    # Delete rar file
    os.system('rm "raw_data/' + folder_name + '/2000_{:s}.rar"'.format(i))

# Delete format.pdf and zip file
os.system('rm raw_data/' + folder_name + '/format.pdf')
os.system('rm raw_data/' + folder_name + '/QKPGroupII.zip')

# %% Download and unzip QKPGroupIII instances

# Download zip file from url
folder_name = 'QKPGroupIII'
url = 'https://leria-info.univ-angers.fr/%7Ejinkao.hao/QKPDATA/QKPGroupIII.zip'
file_name = url.split('/')[-1]
urllib.request.urlretrieve(url, 'raw_data/' + folder_name + '/' + file_name)

# Unzip file
with zipfile.ZipFile('raw_data/' + folder_name + '/' + file_name, 'r') as zip_ref:
    zip_ref.extractall('raw_data')

# Delete zip file
os.system('rm raw_data/' + folder_name + '/QKPGroupIII.zip')

# %% Write standard QKP instances in different format

# Read raw_data
instances_folder = 'Standard-QKP'
n_nodes_list = [100, 200, 300]
densities = {100: [25, 50, 75, 100],
             200: [25, 50, 75, 100],
             300: [25, 50]}
instances = np.arange(1, 11)

for n_nodes in n_nodes_list:
    for density in densities[n_nodes]:
        for instance in instances:

            # Get file name
            file_name = 'jeu_{:d}_{:d}_{:d}'.format(n_nodes, density, instance)

            # Read file
            f = open('raw_data/' + instances_folder + '/' + file_name + '.txt', 'r')
            lines = f.readlines()
            f.close()

            # Get number of nodes and edges
            n_nodes = int(lines[1])

            # Get linear utilities
            edges = []
            linear_utilities = [int(v) for v in lines[2].strip('\n').split(' ') if v != '']
            for i in range(n_nodes):
                edges += [(i, i, linear_utilities[i])]

            # Get quadratic utilities
            for i in range(n_nodes):
                quadratic_utilities = [int(v) for v in lines[3 + i].strip('\n').split(' ') if v != '']
                for j in range(i + 1, n_nodes):
                    edges += [(i, j, quadratic_utilities[j - (i + 1)])]

            # Remove edges with utility of zero
            edges = [edge for edge in edges if edge[2] != 0]

            # Write file
            f = open('collections/' + instances_folder + '/' + file_name + '.txt', 'w')
            f.write('{:d} {:d} {:s}\n'.format(n_nodes, len(edges), 'int'))
            for ind_i, ind_j, val in edges:
                f.write('{:d} {:d} {:.6f}\n'.format(ind_i, ind_j, val))

            # Get weights
            budget = int(lines[4 + n_nodes].strip('\n'))
            weights = [int(v) for v in lines[5 + n_nodes].strip('\n').split(' ') if v != '']

            # Write weights to file
            for weight in weights:
                f.write('{:d} '.format(weight))
            f.write('\n')

            # Write budget to file
            f.write('{:d}\n'.format(budget))
            f.close()

# %% Prepare quadratic knapsack problem instances (from group II)

# Read raw_data
instances_folder = 'QKPGroupII'
n_nodes_list = [1000, 2000]
densities = {1000: [25, 50, 75, 100],
             2000: [25, 50, 75, 100]}
instances = np.arange(1, 11)

for n_nodes in n_nodes_list:
    for density in densities[n_nodes]:
        for instance in instances:

            # Get file name
            file_name = '{:d}_{:d}_{:d}'.format(n_nodes, density, instance)

            # Read file
            f = open('raw_data/' + instances_folder + '/' + file_name + '.dat', 'r')
            lines = f.readlines()
            f.close()

            # Get number of nodes and edges
            n_nodes = int(lines[0])

            # Get linear utilities
            edges = []
            linear_utilities = [int(v) for v in lines[1].strip('\n').split(' ') if v != '']
            for i in range(n_nodes):
                edges += [(i, i, linear_utilities[i])]

            # Get quadratic utilities
            for i in range(n_nodes):
                quadratic_utilities = [int(v) for v in lines[2 + i].strip('\n').split(' ') if v != '']
                for j in range(i + 1, n_nodes):
                    edges += [(i, j, quadratic_utilities[j - (i + 1)])]

            # Remove edges with utility of zero
            edges = [edge for edge in edges if edge[2] != 0]

            # Write file
            f = open('collections/' + instances_folder + '/' + file_name + '.txt', 'w')
            f.write('{:d} {:d} {:s}\n'.format(n_nodes, len(edges), 'int'))
            for ind_i, ind_j, val in edges:
                f.write('{:d} {:d} {:.6f}\n'.format(ind_i, ind_j, val))

            # Get weights
            budget = int(lines[3 + n_nodes].strip('\n'))
            weights = [int(v) for v in lines[4 + n_nodes].strip('\n').split(' ') if v != '']

            # Write weights to file
            for weight in weights:
                f.write('{:d} '.format(weight))
            f.write('\n')

            # Write budget to file
            f.write('{:d}\n'.format(budget))
            f.close()

# %% Prepare quadratic knapsack problem instances (from group III)

# Read raw_data
instances_folder = 'QKPGroupIII'
n_nodes_list = [5000, 6000]
densities = {5000: [25, 50, 75, 100],
             6000: [25, 50, 75, 100]}
instances = np.arange(1, 6)

for n_nodes in n_nodes_list:
    for density in densities[n_nodes]:
        for instance in instances:

            # Get file name
            file_name = '{:d}_{:d}_{:d}'.format(n_nodes, density, instance)

            # Read file
            f = open('raw_data/' + instances_folder + '/' + file_name + '.txt', 'r')
            lines = f.readlines()
            f.close()

            # Get number of nodes and edges
            n_nodes = int(lines[1])

            # Get linear utilities
            edges = []
            linear_utilities = [int(v) for v in lines[3].strip('\n').split(' ') if v != '']
            for i in range(n_nodes):
                edges += [(i, i, linear_utilities[i])]

            # Get quadratic utilities
            for i in range(n_nodes):
                quadratic_utilities = [int(v) for v in lines[4 + i].strip('\n').split(' ') if v != '']
                for j in range(i + 1, n_nodes):
                    edges += [(i, j, quadratic_utilities[j - (i + 1)])]

            # Remove edges with utility of zero
            edges = [edge for edge in edges if edge[2] != 0]

            # Write file
            f = open('collections/' + instances_folder + '/' + file_name + '.txt', 'w')
            f.write('{:d} {:d} {:s}\n'.format(n_nodes, len(edges), 'int'))
            for ind_i, ind_j, val in edges:
                f.write('{:d} {:d} {:.6f}\n'.format(ind_i, ind_j, val))

            # Get weights
            budget = int(lines[6 + n_nodes].strip('\n'))
            weights = [int(v) for v in lines[7 + n_nodes].strip('\n').split(' ') if v != '']

            # Write weights to file
            for weight in weights:
                f.write('{:d} '.format(weight))
            f.write('\n')

            # Write budget to file
            f.write('{:d}\n'.format(budget))
            f.close()

# %% Generate team formation QKP instances from real data sets

max_weight = 10

file_names = os.listdir('raw_data/real team formation data sets/')

for file_name in file_names:

    # Generate utility matrix
    utility_matrix = generate_utility_matrix_from_xlsx_file_of_team_formation_dataset(file_name)

    # Write file
    folder_name = 'TeamFormation-QKP-1'
    f = open('collections/' + folder_name + '/' + file_name.split('_')[0] + '.txt', 'w')
    n_nodes = utility_matrix.shape[0]
    n_edges = int(np.count_nonzero(utility_matrix) / 2)
    f.write('{:d} {:d} {:s}\n'.format(n_nodes, n_edges, 'float'))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if utility_matrix[i, j] > 0:
                f.write('{:d} {:d} {:.6f}\n'.format(i, j, utility_matrix[i, j]))

    # Set random seed
    np.random.seed(24)

    # Generate weights
    weights = np.random.randint(1, max_weight + 1, n_nodes)

    # Add weights
    for weight in weights:
        f.write('{:d} '.format(weight))
    f.write('\n')

    # Add budgets
    for budget_fraction in budget_fractions:
        budget = int(budget_fraction * np.sum(weights))
        f.write('{:d} '.format(budget))

    f.close()

# %% Generate team formation QKP instances from synthetic data sets

max_weight = 10

folder_source = 'raw_data/synthetic team formation data sets/'
file_names = os.listdir(folder_source)

# Sort file_names
file_names.sort()

for i, file_name in enumerate(file_names):

    # Read file
    f = open(folder_source + '/' + file_name, 'r')

    # Read all lines
    lines = f.readlines()

    # Read first line which contains number of nodes and edges
    n_nodes, n_edges = [int(v) for v in lines[0].strip('\n').split(' ')]

    # Check edges
    edges = []
    for j in range(1, n_edges+1):
        ind_i, ind_j, val = lines[j].strip('\n').split(' ')
        if int(ind_i) != int(ind_j):
            edges.append((int(ind_i), int(ind_j), float(val)))

    # Write file
    f = open('collections/TeamFormation-QKP-1/Synthetic_TF_{:d}_n7000.txt'.format(i + 1), 'w')
    f.write('{:d} {:d} {:s}\n'.format(n_nodes, len(edges), 'float'))
    for ind_i, ind_j, val in edges:
        f.write('{:d} {:d} {:.6f}\n'.format(ind_i, ind_j, val))

    # Set random seed
    np.random.seed(24)

    # Generate weights
    weights = np.random.randint(1, max_weight + 1, n_nodes)

    # Add weights
    for weight in weights:
        f.write('{:d} '.format(weight))
    f.write('\n')

    # Add budgets
    for budget_fraction in budget_fractions:
        budget = int(budget_fraction * np.sum(weights))
        f.write('{:d} '.format(budget))

    f.close()
