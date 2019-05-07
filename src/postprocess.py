import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Not enough arguments supplied. Needs 1 (graph name).")
    exit()

graph = sys.argv[1]
file_own_gpu = graph + "_gpu"
file_own_cpu = graph + "_cpu"
file_gunrock = graph + "_gunrock"
file_gapbs = graph + "_gap"

graph = graph.split('/')[-1]

gpu_avg = 0
gpu_min = -1
gpu_max = 0
cpu_avg = 0
cpu_min = -1
cpu_max = 0
gunrock_avg = 0
gunrock_min = 0
gunrock_max = 0
gapbs_avg = 0
gapbs_min = -1
gapbs_max = 0

with open(file_own_gpu, 'r') as gpu:
    count = 0
    for line in gpu:
        count += 1
        if count < 3:
            continue
        
        h2d, kernel, d2h, total = [float(x) for x in line.split(" ")]
        gpu_avg += kernel

        if gpu_max < kernel:
            gpu_max = kernel
        if gpu_min > kernel or gpu_min == -1:
            gpu_min = kernel

    count -= 2
    gpu_avg /= count

with open(file_own_cpu, 'r') as cpu:
    count = 0
    for line in cpu:
        count += 1
        if count < 3:
            continue
        
        h2d, kernel, d2h, total = [float(x) for x in line.split(" ")]
        cpu_avg += kernel

        if cpu_max < kernel:
            cpu_max = kernel
        if cpu_min > kernel or cpu_min == -1:
            cpu_min = kernel

    count -= 2
    cpu_avg /= count

with open(file_gapbs, 'r') as gap:
    for line in gap:
        if line.startswith("Average Time"):
            gapbs_avg = float(line.split(" ")[-1]) * 1000
        elif line.startswith("Trial Time"):
            val = float(line.split(" ")[-1]) * 1000
            if gapbs_max < val:
                gapbs_max = val
            if gapbs_min > val or gapbs_min == -1:
                gapbs_min = val

with open(file_gunrock, 'r') as gunrock:
    for line in gunrock:
        if line.startswith(" avg. elapsed"):
            gunrock_avg = float(line.split(" ")[-2])
        if line.startswith(" min. elapsed"):
            gunrock_min = float(line.split(" ")[-2])
        if line.startswith(" max. elapsed"):
            gunrock_max = float(line.split(" ")[-2])

plt.bar([0, 1, 2, 3], 
    [gunrock_avg, gpu_avg, cpu_avg, gapbs_avg], 
    yerr=[[gunrock_max - gunrock_avg, gpu_max - gpu_avg, cpu_max - cpu_avg, gapbs_max - gapbs_avg], [gunrock_avg - gunrock_min, gpu_avg - gpu_min, cpu_avg - cpu_min, gapbs_avg - gapbs_min]],
    ecolor='red',
    tick_label=['Gunrock (gpu)', 'Rodinia gpu', 'Rodinia cpu', 'GAP (cpu)'])
plt.xlabel("Platform and implementation")
plt.ylabel("Execution time (ms)")
plt.title("Execution time for unoptimized Rodinia\n compared to Gunrock and GAP on" + graph)
#plt.show()
plt.savefig("exec_" + graph + ".pdf")


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.bar([0, 1], 
    [100, gunrock_avg / gpu_avg * 100], 
    yerr=[[0, 0], [0, 0]],
    tick_label=['Gunrock (gpu)', 'Rodinia gpu'])
ax1.set_ylabel("Application Efficiency (%)")

ax2.bar([0, 1], 
    [100, gapbs_avg / cpu_avg * 100], 
    yerr=[[0, 0], [0, 0]],
    tick_label=['GAP (cpu)', 'Rodinia cpu'])
fig.suptitle("Application Efficiency for unoptimized Rodinia\n compared to Gunrock and GAP on " + graph)
plt.savefig("appeff_" + graph + ".pdf")
#plt.show()