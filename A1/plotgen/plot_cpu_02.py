import pandas as pd
import matplotlib.pyplot as plt

# CSV file exported from your STREAM output
csv_file = "stream_cpu_O2.csv"

# Load CSV
df = pd.read_csv(csv_file)

plt.figure(figsize=(8,5))
plt.plot(df['N'], df['GB_per_s'], marker='o', label='O2 Scalar')
plt.xscale('log')
plt.xlabel('Array size N')
plt.ylabel('Memory bandwidth (GB/s)')
plt.title('STREAM Triad: CPU O2 Scalar')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_O2_bandwidth.png")
plt.show()
