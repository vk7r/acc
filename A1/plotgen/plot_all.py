import re
import matplotlib.pyplot as plt

def parse_outfile(filename):
    sizes, gbps, mupds = [], [], []
    with open(filename, "r") as f:
        for line in f:
            match = re.search(
                r"STREAM triad of size\s+(\d+).*?([\d\.]+)\s+MUPD/s.*?([\d\.]+)\s+GB/s",
                line
            )
            if match:
                sizes.append(int(match.group(1)))
                mupds.append(float(match.group(2)))
                gbps.append(float(match.group(3)))
    return sizes, gbps, mupds


# -----------------------------
# Task 2: CPU O2 vs O3
# -----------------------------
# cpu_opt_files = {
#     "Personal CPU 03" : "cpu_native_results.out",
#     # "CPU O2": "cpu_02_results.out",
#     # "CPU O3": "cpu_03_results.out"
# }

# plt.figure(figsize=(8,6))
# for label, fname in cpu_opt_files.items():
#     sizes, gbps, _ = parse_outfile(fname)
#     plt.plot(sizes, gbps, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Memory Bandwidth (GB/s)")
# plt.title("Task 3: CPU STREAM Triad O3 -  Different Hardware")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task3_cpu_03_personal.png", dpi=300)
# plt.close()


# -----------------------------
# Task 4: SIMD aligned vs unaligned
# -----------------------------
cpu_simd_files = {
    "CPU Aligned SIMD": "plotgen/cpu_aligned_results.out",
    "CPU Unaligned SIMD": "plotgen/cpu_unaligned_results.out",
    "i5-1135G7 unaligned" : "plotgen/unaligned_personal_cpu.out",
    "i5-1135G7 aligned" : "plotgen/aligned_personal_cpu.out",
}

plt.figure(figsize=(8,6))
for label, fname in cpu_simd_files.items():
    sizes, gbps, _ = parse_outfile(fname)
    plt.plot(sizes, gbps, marker="o", label=label)

plt.xscale("log")
plt.xlabel("Vector Size N")
plt.ylabel("Memory Bandwidth (GB/s)")
plt.title("Task 4: i5-1135G7 (Aligned vs Unaligned)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("task4_cpu_PERSONALasdads_simd.png", dpi=300)
plt.close()

# # -----------------------------
# # Task 5 & 6: GPU block sizes
# # -----------------------------
# gpu_small_files = {
#     "Block size 1": "cuda_results_1.out",
#     "Block size 64": "cuda_results_64.out",
#     "Block size 128": "cuda_results_128.out",
#     "Block size 256": "cuda_results_256.out"
# }

# gpu_mid_files = {
#     "Block size 512 (default)": "cuda_results_default_512.out",
#     "Block size 1024": "cuda_results_1024.out"
# }

# gpu_extreme_files = {
#     "Block size 2048": "cuda_results_2048.out"
# }

# # Small block sizes
# plt.figure(figsize=(8,6))
# for label, fname in gpu_small_files.items():
#     sizes, gbps, _ = parse_outfile(fname)
#     plt.plot(sizes, gbps, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Memory Bandwidth (GB/s)")
# plt.title("Task 5 & 6: GPU STREAM Triad (Small Block Sizes: 1–256)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task5_6_gpu_small.png", dpi=300)
# plt.close()

# # Mid block sizes (512 and 1024)
# plt.figure(figsize=(8,6))
# for label, fname in gpu_mid_files.items():
#     sizes, gbps, _ = parse_outfile(fname)
#     plt.plot(sizes, gbps, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Memory Bandwidth (GB/s)")
# plt.title("Task 5 & 6: GPU STREAM Triad (Block Sizes 512 & 1024)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task5_6_gpu_mid.png", dpi=300)
# plt.close()

# # Extreme block size (2048 alone)
# plt.figure(figsize=(8,6))
# for label, fname in gpu_extreme_files.items():
#     sizes, gbps, _ = parse_outfile(fname)
#     plt.plot(sizes, gbps, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Memory Bandwidth (GB/s)")
# plt.title("Task 5 & 6: GPU STREAM Triad (Block Size 2048)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task5_6_gpu_extreme.png", dpi=300)
# plt.close()


# # -----------------------------
# # Task 7: GPU single vs double precision
# # -----------------------------
# precision_files = {
#     "GPU Single Precision (512)": "cuda_results_default_512.out",
#     "GPU Double Precision (512)": "cuda_result_default_DOUBLE_PRECISION.out"
# }

# # GB/s comparison
# plt.figure(figsize=(8,6))
# for label, fname in precision_files.items():
#     sizes, gbps, _ = parse_outfile(fname)
#     plt.plot(sizes, gbps, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Memory Bandwidth (GB/s)")
# plt.title("Task 7: GPU STREAM Triad (Single vs Double Precision, GB/s)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task7_gpu_precision_gbps.png", dpi=300)
# plt.close()

# # MUPD/s comparison
# plt.figure(figsize=(8,6))
# for label, fname in precision_files.items():
#     sizes, _, mupds = parse_outfile(fname)
#     plt.plot(sizes, mupds, marker="o", label=label)

# plt.xscale("log")
# plt.xlabel("Vector Size N")
# plt.ylabel("Throughput (MUPD/s)")
# plt.title("Task 7: GPU STREAM Triad (Single vs Double Precision, MUPD/s)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig("task7_gpu_precision_mupd.png", dpi=300)
# plt.close()

