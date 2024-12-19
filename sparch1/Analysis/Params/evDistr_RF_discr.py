import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sympy import symbols, Eq, solve, sqrt, pi

# Adjust the path to include the project_root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sparch.models.snns import SNN
from sparch.models.snns import LIFcomplexLayer, adLIFclampLayer, BRFLayer, ResonateFireLayer

dataset_name = "shd"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name == "shd" else 35
neuron_model = 'ResonateFire'
nb_hiddens = 512
nb_layers = 3

layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
input_shape = (batch_size, None, nb_inputs)

extra_features = {
    'superspike': False,
    'slayer': False,
    'xavier_init': False,
    'no_reset': False,
    'half_reset': True,
    'rst_detach': False,
    'dt_min': 0.01,
    'dt_max': 0.4,
    's_GLU': False,
    'time_offset': 10,
    'thr': 1,
    'bRand': 'Rand',
    'c_discr': False,
    'c_param': False,
    'reset': 'half_reset', 
    'recurrent': False
}

net = SNN(
    input_shape=input_shape,
    layer_sizes=layer_sizes,
    neuron_type=neuron_model,
    dropout=0.1,
    normalization="batchnorm",
    use_bias=False,
    bidirectional=False,
    use_readout_layer=True,
    extra_features=extra_features
)

# Load the trained model
model_path = '../../SHD_runs/exp/paper_models/shd_ResonateFire_3lay512_/checkpoints/best_model.pth'


trained_net = torch.load(model_path)

def plot_eigenvalue_distribution():
    R = 1  # You may adjust this value for different systems
    tau_m_values = np.linspace(-0.1, 1.2, 500) 
    a_R_values = np.linspace(-1.2, 1.2, 500)

    # Set up the plot
    plt.figure(figsize=(8, 6))
    plt.gca().add_patch(Rectangle((0.0196, -1), 1, 2, color='gray', alpha=0.7, zorder=2))

    for tau_m_over_tau_w in tau_m_values:
        for a_R in a_R_values:
            tau_m = 1  # Arbitrary tau_m value
            tau_w = tau_m / tau_m_over_tau_w
            a_w = a_R / R
            eigenvalues, _ = find_system_eigenvalues_numeric(tau_m, tau_w, R, a_w)
            
            # Compute the Nullstelle (1/tau_w)
            nullstelle = -1 / tau_w
            eigenvalues = [eigenvalues[0][0],eigenvalues[1][1]]
            if (eigenvalues[0]>0 and eigenvalues[0]!=nullstelle) or (eigenvalues[1]>0 and eigenvalues[1]!=nullstelle):
                plt.plot(tau_m_over_tau_w, a_R, 'gs', alpha=0.2, zorder=1)
            # If eigenvalues are complex
            elif np.iscomplex(eigenvalues).any():
                plt.plot(tau_m_over_tau_w, a_R, 'bs', alpha=0.2, zorder=1) # Blue dots for complex
            else:
                # Compare Nullstelle with min(eigenvalue)
                if nullstelle > eigenvalues[0] and nullstelle > eigenvalues[1]:
                    plt.plot(tau_m_over_tau_w, a_R, 'rs', alpha=0.2, zorder=1)  # Red dots if Nullstelle < min(eigenvalue)
                else:
                    plt.plot(tau_m_over_tau_w, a_R, 'ys', alpha=0.2, zorder=1) # Yellow dots if Nullstelle >= min(eigenvalue)

    # Create custom legend handles with correct colors
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, alpha=1, label='Oscillatory Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, alpha=1, label='Non-oscillatory (Mixed) Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, alpha=1, label='Non-oscillatory (integrator) Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, alpha=1, label='Unstable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, alpha=1, label='Neuron parameters after training'),
        Rectangle((0, 0), 1, 1, color='gray', alpha=0.9, label='Sparch AdLIF region')
    ]

    # Add legend
    plt.legend( handles=legend_elements)

    # Add labels and title
    

def find_system_eigenvalues_numeric(tau_m_array, tau_w_array, R_array, a_w_array):
    eigenvalue_list = []

    # Assuming tau_m_array, tau_w_array, R_array, a_w_array are all numpy arrays
    for tau_m, tau_w, R, a_w in zip(tau_m_array, tau_w_array, R_array, a_w_array):
        # Compute the eigenvalue for each set of parameters.
        # Modify this part according to how you compute eigenvalues.
        # For demonstration, let's assume a 2D system with a matrix A whose eigenvalues we're finding.
        
        # Define the system matrix based on tau_m, tau_w, R, a_w
        A = np.array([[-1/tau_m, -R/tau_m], [a_w/tau_w, -1/tau_w]])  # Example system matrix
        
        # Compute eigenvalues of the matrix
        eigenvalues = np.linalg.eigvals(A)
        
        # Store eigenvalues
        eigenvalue_list.append(eigenvalues)
    
    return np.array(eigenvalue_list) 

def compute_alpha(log_log_alpha, alpha_img, log_dt):
    alpha = torch.exp((-torch.exp(log_log_alpha) + 1j * alpha_img) * torch.exp(log_dt))
    return alpha

def compute_rat_R_a(real_value, img_value, tau_m=0.01):
    rat, R_a = symbols('rat R_a')
    eq1 = Eq(-(rat + 1) / (2 * tau_m), real_value)
    eq2 = Eq(sqrt((rat**2 - 2*rat + 1 - 4*R_a * rat)*(-1)) / (2 * tau_m), img_value)
    solutions = solve([eq1, eq2], (rat, R_a))
    return solutions

print(trained_net)
# Main Plot Function
for i, trained_layer in enumerate(trained_net.snn):
    print(trained_layer)
    if isinstance(trained_layer, ResonateFireLayer):
        
        alpha_im = trained_layer.alpha_im.detach().cpu().numpy()
        alpha_real = trained_layer.alpha_real.detach().cpu().numpy()
        alpha_real = np.clip(alpha_real, max = -0.1)
        dt = 0.004

        discreteEV = np.exp((alpha_real+1j*alpha_im)*dt)

        

        # Create the scatter plot with different colors for different conditions
        plt.figure(figsize=(8, 6))

        # Oscillatory Dynamics (blue)
        plt.scatter(discreteEV.real, discreteEV.imag, color='blue', marker='x', label='Oscillatory Dynamics', s=30, alpha = 0.9)

        # Label axes
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')

        # Title and grid
        plt.title('Scatter Plot of Eigenvalues on Complex Plane (Dynamics Categories)')
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.grid(True)

        # Create custom legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, alpha=1, label='Oscillatory Dynamics'),]

        # Add legend to the plot
        plt.legend(handles=legend_elements, loc='upper left')

        plt.show()

        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.title(f'subthreshold parameters of sparch AdLIF for layer {i}')
        plt.grid(True)

        plt.savefig(f"../../plots/SHD/EV/discrRFEVs_layer{i}.png")
        plt.show()
        plt.clf()  # Clear the figure for the next plot

