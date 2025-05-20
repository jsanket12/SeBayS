import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from Excel file
file_path = './Plots/BayS_models.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Extract columns
M = data['M'].to_list()
individual_learners = np.array(data['Ind_Acc_Mean'].to_list())
individual_learners_std = np.array(data['Ind_Acc_Std'].to_list())
ensemble = np.array(data['Ens_Acc'].to_list())
print(ensemble)

fontsize = 28
# Plotting
plt.figure(figsize=(8.15, 8))
# plt.errorbar(M, individual_learners, yerr=individual_learners_std, color='blue', ls = "None", label='Individual Learners')
plt.plot(M, individual_learners, '^--b', lw=2, ms=15, label='Individual Learners', zorder=2)
plt.fill_between(M, individual_learners - individual_learners_std, individual_learners + individual_learners_std, color= 'blue', lw=0, alpha=0.3, zorder=2)
plt.plot(M, ensemble, '^-g', lw=2, ms=15, label='BayS Ensemble', zorder=2)

plt.xlabel('M', fontsize=fontsize)
plt.ylabel('Test Accuracy (%)', fontsize=fontsize)
plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=0.8*fontsize)
plt.yticks([95,95.25,95.5,95.75,96,96.25,96.50,96.75],fontsize=0.8*fontsize)
plt.ylim(95,96.75)
plt.xlim(0.5, 10.5)
plt.legend()
plt.grid(True, alpha=0.4, zorder=1)

plt.tight_layout()
legend = plt.legend(fontsize=0.8*fontsize, loc='upper right')
legend.set_zorder(3)
plt.savefig('./Plots/BayS_Ensemble_Size_Effect.pdf',dpi=300)
plt.savefig('./Plots/BayS_Ensemble_Size_Effect.png')
plt.show()