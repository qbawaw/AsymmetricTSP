import pandas as pd
import matplotlib.pyplot as plt

held_karp_data = pd.read_csv('held_karp_times.csv')
rnn_data = pd.read_csv('rnn_times.csv')
aco_data = pd.read_csv('aco_times.csv')

plt.figure(figsize=(8, 6))
plt.plot(held_karp_data['Size'], held_karp_data['Held-Karp Time'], color='blue', linestyle='-', linewidth=2)
plt.title('Held-Karp', fontsize=16)
plt.xlabel('Liczba wierzchołków', fontsize=14)
plt.ylabel('Czas działania (s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('held_karp_plot.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(rnn_data['Size'], rnn_data['RNN Time'], color='green', linestyle='-', linewidth=2)
plt.title('Repetitive Nearest Neighbour', fontsize=16)
plt.xlabel('Liczba wierzchołków', fontsize=14)
plt.ylabel('Czas działania (s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('rnn_plot.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(aco_data['Size'], aco_data['ACO Time'], color='red', linestyle='-', linewidth=2)
plt.title('Ant Colony Optimization', fontsize=16)
plt.xlabel('Liczba wierzchołków', fontsize=14)
plt.ylabel('Czas działania (s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('aco_plot.png')
plt.show()