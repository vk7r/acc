import pandas as pd
import matplotlib.pyplot as plt

csv_file = "stream_cpu_simd.csv"
df = pd.read_csv(csv_file)

plt.figure(figsize=(8,5))
plt.plot(df['N'], df['GB_per_s'], marker='o', label='SIMD unaligned')
plt.plot(df['N'], df['GB_per_s_aligned'], marker='s', label='SIMD aligned')
plt.xscale('log')
plt.xlabel('Array size N')
plt.ylabel('Memory bandwidth (GB/s)')
plt.title('STREAM Triad: CPU SIMD Alignment')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_simd_bandwidth.png")
plt.show()
