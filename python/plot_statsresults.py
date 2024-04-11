# Visualize the results of the statistical analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
#file_path = '../results/statresults_comb.csv'
file_path = '../results/tsai/20240410_163628/results_stats_all.csv'
data = pd.read_csv(file_path)
output_path = '../results/tsai/20240410_163628/'

# Display the first few rows of the dataframe
data.head()

# Create a new column that combines target, wavefront, and method for labeling purposes
data['target_wavefront_method'] = data['target'] + "_" + data['wavefront'] + "_" + data['method']

# Plot and rank a barchart for accuracy and AUC for all entries
fig, ax = plt.subplots(2, 1, figsize=(12, 16))

# Sort data by accuracy and AUC for plotting
sorted_by_accuracy = data.sort_values(by='accuracy', ascending=False)
sorted_by_auc = data.sort_values(by='auc', ascending=False)

# Accuracy plot
sns.barplot(x='accuracy', y='target_wavefront_method', data=sorted_by_accuracy, ax=ax[0], palette="viridis")
ax[0].set_title('Ranking by Accuracy')
ax[0].set_xlabel('Accuracy')
ax[0].set_ylabel('Target_Wavefront_Method')

# AUC plot
sns.barplot(x='auc', y='target_wavefront_method', data=sorted_by_auc, ax=ax[1], palette="magma")
ax[1].set_title('Ranking by AUC')
ax[1].set_xlabel('AUC')
ax[1].set_ylabel('')

plt.tight_layout()
plt.savefig(output_path + 'ranked_barcharts.png', dpi =200)
plt.show()



# Group by 'target' and calculate mean accuracy and AUC for each group
grouped_by_target_accuracy = data.groupby('target')['accuracy'].mean().sort_values(ascending=False).reset_index()
grouped_by_target_auc = data.groupby('target')['auc'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot for mean accuracy by target
sns.barplot(x='accuracy', y='target', data=grouped_by_target_accuracy, ax=ax[0], palette="viridis")
ax[0].set_title('Mean Accuracy by Target')
ax[0].set_xlabel('Mean Accuracy')
ax[0].set_ylabel('Target')

# Plot for mean AUC by target
sns.barplot(x='auc', y='target', data=grouped_by_target_auc, ax=ax[1], palette="magma")
ax[1].set_title('Mean AUC by Target')
ax[1].set_xlabel('Mean AUC')
ax[1].set_ylabel('Target')

plt.tight_layout()
plt.savefig(output_path + 'mean_accuracy_auc_by_target.png',dpi=200)
plt.show()



# Group by 'wavefront' and calculate mean accuracy and AUC for each group
grouped_by_wavefront_accuracy = data.groupby('wavefront')['accuracy'].mean().sort_values(ascending=False).reset_index()
grouped_by_wavefront_auc = data.groupby('wavefront')['auc'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot for mean accuracy by wavefront
sns.barplot(x='accuracy', y='wavefront', data=grouped_by_wavefront_accuracy, ax=ax[0], palette="viridis")
ax[0].set_title('Mean Accuracy by Wavefront')
ax[0].set_xlabel('Mean Accuracy')
ax[0].set_ylabel('Wavefront')

# Plot for mean AUC by wavefront
sns.barplot(x='auc', y='wavefront', data=grouped_by_wavefront_auc, ax=ax[1], palette="magma")
ax[1].set_title('Mean AUC by Wavefront')
ax[1].set_xlabel('Mean AUC')
ax[1].set_ylabel('Wavefront')

plt.tight_layout()
plt.savefig(output_path + 'mean_accuracy_auc_by_wavefront.png',dpi=200)
plt.show()


# Group by 'method' and rank with regard to accuracy and AUC

# Group by 'method' and calculate mean accuracy and AUC for each group
grouped_by_method_accuracy = data.groupby('method')['accuracy'].mean().sort_values(ascending=False).reset_index()
grouped_by_method_auc = data.groupby('method')['auc'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Plot for mean accuracy by method
sns.barplot(x='accuracy', y='method', data=grouped_by_method_accuracy, ax=ax[0], palette="viridis")
ax[0].set_title('Mean Accuracy by Method')
ax[0].set_xlabel('Mean Accuracy')
ax[0].set_ylabel('Method')

# Plot for mean AUC by method
sns.barplot(x='auc', y='method', data=grouped_by_method_auc, ax=ax[1], palette="magma")
ax[1].set_title('Mean AUC by Method')
ax[1].set_xlabel('Mean AUC')
ax[1].set_ylabel('Method')

plt.tight_layout()
plt.savefig(output_path + 'mean_accuracy_auc_by_method.png',dpi=200)
plt.show()