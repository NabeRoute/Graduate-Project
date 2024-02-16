import pandas as pd
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PM
from joblib import load
import veegan_new
import veegan
import torch

# Load and process dataset
dataset_df = pd.read_excel('./data/MPEA_parsed_dataset.xlsx')
element_names = dataset_df.columns.values[14:46]
process_names = dataset_df.columns.values[46:53]
feature_names = dataset_df.columns.values[14:53]

data_np = dataset_df.to_numpy()
comp_data = data_np[:, 14:53].astype(float)
comp_min = np.min(comp_data, axis=0)
comp_max = np.max(comp_data, axis=0)

# Load pre-trained generator and evaluators
#generator = veegan_new.Generator()
generator = veegan_new.Generator()
generator.load_state_dict(torch.load('VEEGAN/VEEGAN_experiment_9000epocs_adam.pt'))
generator.eval()

hard_regressor = load('./saved_models/hardness_regressor.joblib')
yield_regressor = load('./saved_models/yield_regressor.joblib')
tensile_regressor = load('./saved_models/tensile_regressor.joblib')
elongation_regressor = load('./saved_models/elongation_regressor.joblib')

classifier_FCC = load('./saved_models/FCC_classifier.joblib')
classifier_BCC = load('./saved_models/BCC_classifier.joblib')
classifier_HCP = load('./saved_models/HCP_classifier.joblib')
classifier_IM = load('./saved_models/IM_classifier.joblib')

# Define the optimization problem for alloy design
class AlloyOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=-3, xu=3)

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            fake_alloys = generator(x_tensor).numpy()

        fake_alloys = (fake_alloys * comp_max) + comp_min

        f1 = - hard_regressor.predict(fake_alloys)
        f2 = - elongation_regressor.predict(fake_alloys)
        #f3 = - phase_regressor.predict(fake_alloys)
        out["F"] = np.column_stack([f1, f2])


# Run optimization
problem = AlloyOptimizationProblem()

algorithm = NSGA2(pop_size=80, mutation=PM(prob=0.1, eta=30))
termination = get_termination("n_gen", 800)

res = minimize(problem, algorithm, termination, pf=problem.pareto_front(), save_history=True, seed=3, verbose=False)
#np.save('NSGA2_FM_GP_history.npy', res.F)

# Post-process results to compute optimal alloys
result_tensor = torch.tensor(res.X, dtype=torch.float32)
with torch.no_grad():
    optimal_alloys = generator(result_tensor).numpy()
optimal_alloys = (optimal_alloys * comp_max) + comp_min

# Normalize molar ratios
optimal_alloys[:, :32] /= np.sum(optimal_alloys[:, :32], axis=1).reshape((-1, 1))

# Generate alloy names based on composition
alloy_names = []
for i in range(optimal_alloys.shape[0]):
    composition = optimal_alloys[i, :32]
    alloy_name = "".join(
        [element_names[j] + str(round(composition[j], 3)) for j in range(len(composition)) if composition[j] > 0.005])
    alloy_names.append(alloy_name)

# Convert process index to one-hot encoding
process_indices = np.argmax(optimal_alloys[:, 32:], axis=1)
optimal_alloys[:, 32:] = np.eye(optimal_alloys.shape[1] - 32)[process_indices]

# Calculate properties for optimal alloys
property_array = np.column_stack([
    elongation_regressor.predict(optimal_alloys),
    tensile_regressor.predict(optimal_alloys),
    yield_regressor.predict(optimal_alloys),
    hard_regressor.predict(optimal_alloys)
])

phase_array = np.zeros((optimal_alloys.shape[0], 4))
phase_array[:, 0] = classifier_FCC.predict(optimal_alloys)
phase_array[:, 1] = classifier_BCC.predict(optimal_alloys)
phase_array[:, 2] = classifier_HCP.predict(optimal_alloys)
phase_array[:, 3] = classifier_IM.predict(optimal_alloys)

phase_name_list = []
phase_name_strings = ['FCC', 'BCC', 'HCP', 'IM']
for i in range(optimal_alloys.shape[0]):
    phase_name = ''
    for j in range(4):
        if phase_array[i][j] >0:
            phase_name += phase_name_strings[j]
            phase_name += '+'
    phase_name_list.append(phase_name[:-1])

process_name_list = [process_names[i] for i in process_indices]

# Convert data to dataframe and export
formula_df = pd.DataFrame(zip(alloy_names, process_name_list, phase_name_list), columns=['Composition', 'Processing method', 'Predicted phase'])
property_df = pd.DataFrame(property_array, columns=['Elongation', 'Tensile', 'Yield', 'Hardness'])
feature_df = pd.DataFrame(optimal_alloys, columns=feature_names)

output_df = pd.concat([formula_df, property_df, feature_df], axis=1)
output_df.to_excel('VEEGAN/NSGA2_generated/NSGA2_phase_9000epocs_model.xlsx', index=False)

gene_df = pd.DataFrame(res.X, columns=['gene{}'.format(i) for i in range(1, 11)])
gene_mapping_df = pd.concat([gene_df, formula_df, property_df], axis=1)
gene_mapping_df.to_excel('VEEGAN/NSGA2_generated/NSGA2_phase_mapping_9000epocs_model.xlsx', index=False)