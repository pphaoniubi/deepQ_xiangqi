import pandas as pd
import matplotlib.pyplot as plt

# Load the log file
df = pd.read_csv('training_log.csv')

# If needed: remove duplicate header rows (some logs might accidentally repeat the header)
df = df[df['Iteration'] != 'Iteration']
df['Iteration'] = df['Iteration'].astype(int)
df['Total Loss'] = df['Total Loss'].astype(float)
df['Policy Loss'] = df['Policy Loss'].astype(float)
df['Value Loss'] = df['Value Loss'].astype(float)

# Optional: Smooth using rolling average
window = 100
df['Total Loss (Smoothed)'] = df['Total Loss'].rolling(window).mean()
df['Policy Loss (Smoothed)'] = df['Policy Loss'].rolling(window).mean()
df['Value Loss (Smoothed)'] = df['Value Loss'].rolling(window).mean()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['Iteration'], df['Total Loss (Smoothed)'], label='Total Loss')
plt.plot(df['Iteration'], df['Policy Loss (Smoothed)'], label='Policy Loss')
plt.plot(df['Iteration'], df['Value Loss (Smoothed)'], label='Value Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations (Smoothed)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Read CSV
df = pd.read_csv('training_log.csv')

# Plot policy entropy
plt.plot(df['step'], df['policy_entropy'])
plt.xlabel('Step')
plt.ylabel('Policy Entropy')
plt.title('Policy Entropy over Training')
plt.show()

# Plot loss
plt.plot(df['step'], df['loss'], label='Total loss')
plt.plot(df['step'], df['loss_policy'], label='Policy loss')
plt.plot(df['step'], df['loss_value'], label='Value loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss components over Training')
plt.show()