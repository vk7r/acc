import pandas as pd
import matplotlib.pyplot as plt

csv_file = "stream_cpu_O3.csv"
df = pd.read_csv(csv_file)

plt.figure(figsize=(8,5))
plt.plot(df['N'], df['GB_per_s'], marker='o', color='green', label='O3 Vectorized')
plt.xscale('log')
plt.xlabel('Array size N')
plt.ylabel('Memory bandwidth (GB/s)')
plt.title('STREAM Triad: CPU O3 Vectorized')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_O3_bandwidth.png")
plt.show()
