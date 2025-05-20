
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load data from Excel file
file_path = './Plots/BayS_sparsity_effect_cifar100.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Extract columns
# Sparsity = data['Sparsity'].to_list()
Sparsity = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
individual_learners = np.array(data['Ind_Acc_Mean'].to_list())
individual_learners_std = np.array(data['Ind_Acc_Std'].to_list())
ensemble = np.array(data['Ens_Acc'].to_list())
print(np.array(data['Ens_Acc'].to_list()))
KL = np.array(data['KL'].to_list())
print(np.array(data['KL'].to_list()))

# Create x-axis positions using indices
x = np.arange(len(Sparsity))

fontsize = 39
# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(14.5, 12))

# Create a twin axis sharing the same x-axis for KL values
ax2 = ax1.twinx()
x = np.arange(len(Sparsity))
ax1.plot(x, individual_learners, '^--b', lw=2, ms=15, label='Individual Learners', zorder=2)
ax1.fill_between(x, individual_learners - individual_learners_std, 
                 individual_learners + individual_learners_std, color='blue', lw=0, alpha=0.3, zorder=2)
ax1.plot(x, ensemble, '^-g', lw=2, ms=15, label='BayS Ensemble', zorder=4)
ax2.plot(x, KL, '^-r', lw=2, ms=15, label='KL Divergence', zorder=3)

# Label the axes
ax1.set_xlabel('Sparsity', fontsize=fontsize)
ax1.set_ylabel('Test Accuracy (%)', fontsize=fontsize)
ax2.set_ylabel('KL', fontsize=fontsize)

# Set x-ticks and custom labels
ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in Sparsity], fontsize=0.8 * fontsize)
ax1.set_xlim(-0.25, len(Sparsity) - 0.75)

# Set left y-axis ticks and limits for accuracy
# ax1.set_yticks([95.25, 95.5, 95.75, 96, 96.25, 96.50, 96.75])
# ax1.tick_params(axis='y', labelsize=0.8 * fontsize)
# ax1.set_ylim(95.15, 96.85)
# ax2.set_yticks([0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120]) # ([0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.100]) 
# ax2.tick_params(axis='y', labelsize=0.8 * fontsize)
# ax2.set_ylim(0.088, 0.122)

ax1.set_yticks([78.25, 79, 79.75, 80.5, 81.25, 82, 82.75, 83.5]) # ([79, 79.5, 80, 80.5, 81, 81.5, 82, 82.5])
ax1.tick_params(axis='y', labelsize=0.8 * fontsize)
ax1.set_ylim(77.95, 83.8) # (78.8, 82.7)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.set_yticks([0.360, 0.400, 0.440, 0.480, 0.520, 0.560, 0.600, 0.640]) # ([0.275, 0.280, 0.285, 0.290, 0.295, 0.300, 0.305, 0.310])
ax2.tick_params(axis='y', labelsize=0.8 * fontsize)
ax2.set_ylim(0.344, 0.656) # (0.273, 0.312) 
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

# Optionally combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=0.8 * fontsize)

# Add grid and tighten layout
ax1.grid(True, alpha=0.4, zorder=1)
ax2.grid(True, alpha=0.4, zorder=1)
plt.tight_layout()

# Save figures
plt.savefig('./Plots/BayS_Sparsity_Effect_Cifar100.pdf', dpi=300)
plt.savefig('./Plots/BayS_Sparsity_Effect_Cifar100.png')

# Show plot
plt.show()