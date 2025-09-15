import pandas as pd
import matplotlib.pyplot as plt

csv_file = "stream_gpu_precision.csv"
df = pd.read_csv(csv_file)

plt.figure(figsize=(8,5))
plt.plot(df['N'], df['GB_per_s_single'], marker='o', label='Single Precision')
plt.plot(df['N'], df['GB_per_s_double'], marker='s', label='Double Precision')
plt.xscale('log')
plt.xlabel('Array size N')
plt.ylabel('Memory bandwidth (GB/s)')
plt.title('STREAM Triad: GPU Single vs Double Precision')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("gpu_precision_bandwidth.png")
plt.show()
