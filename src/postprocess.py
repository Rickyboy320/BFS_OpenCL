import sys
import matplotlib.pyplot as plt

def read_file(file_read):
    out_min = 0
    out_avg = 0
    out_max = 0
   
    with open(file_read, 'r') as gpu:
    count = 0
    for line in gpu:
        count += 1
        if count < 3:
            continue
        
        h2d, kernel, d2h, total = [float(x) for x in line.split(" ")]
        out_avg += kernel

        if out_max < kernel:
            out_max = kernel
        if out_min > kernel or out_min == -1:
            out_min = kernel

    count -= 2
    out_avg /= count

    return out_min, out_avg, out_max



if len(sys.argv) < 2:
    print("Not enough arguments supplied. Needs 1 (graph name).")
    exit()

graph = sys.argv[1]
file_own_gpu = graph + "_gpu"
file_own_cpu = graph + "_cpu"
file_dir_gpu = graph + "_dir_gpu"
file_dir_cpu = graph + "_dir_cpu"

file_dat_gpu = graph + "_dat_gpu"
file_dat_cpu = graph + "_dat_cpu"

file_dat_dir_gpu = graph + "_dat_dir_gpu"
file_dat_dir_cpu = graph + "_dat_dir_cpu"

file_gunrock = graph + "_gunrock"
file_gapbs = graph + "_gap"

graph = graph.split('/')[-1]

gpu_min, gpu_avg, gpu_max = read_file(file_own_gpu)
cpu_min, cpu_avg, cpu_max = read_file(file_own_cpu)

gpu_dir_min, gpu_dir_avg, gpu_dir_max = read_file(file_dir_gpu)
cpu_dir_min, cpu_dir_avg, cpu_dir_max = read_file(file_dir_cpu)

gpu_dat_min, gpu_dat_avg, gpu_dat_max = read_file(file_dat_gpu)
cpu_dat_min, cpu_dat_avg, cpu_dat_max = read_file(file_dat_cpu)

gpu_dat_dir_min, gpu_dat_dir_avg, gpu_dat_dir_max = read_file(file_dat_dir_gpu)
cpu_dat_dir_min, cpu_dat_dir_avg, cpu_dat_dir_max = read_file(file_dat_dir_cpu)

gunrock_avg = 0
gunrock_min = 0
gunrock_max = 0

gapbs_avg = 0
gapbs_min = -1
gapbs_max = 0

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
    [gunrock_avg, gpu_avg, gpu_dat_avg, gpu_dir_avg, gpu_dat_dir_avg, gapbs_avg, cpu_avg, cpu_dat_avg, cpu_dir_avg, cpu_dat_dir_avg], 
    yerr=[[gunrock_max - gunrock_avg, gpu_max - gpu_avg, gpu_dat_max - gpu_dat_avg, gpu_dir_max - gpu_dir_avg, gpu_dat_dir_max - gpu_dat_dir_avg,
           gapbs_max - gapbs_avg, cpu_max - cpu_avg, cpu_dat_max - cpu_dat_avg, cpu_dir_max - cpu_dir_avg, cpu_dat_dir_max - cpu_dat_dir_avg], 
          [gunrock_avg - gunrock_min, gpu_avg - gpu_min, gpu_dat_avg - gpu_dat_min, gpu_dir_avg - gpu_dir_min, gpu_dat_dir_avg - gpu_dat_dir_min,
           gapbs_avg - gapbs_min, cpu_avg - cpu_min, cpu_dat_avg - cpu_dat_min, cpu_dir_avg - cpu_dir_min, cpu_dat_dir_avg - cpu_dat_dir_min]],
    ecolor='red',
    tick_label=['Gunrock (gpu)', 'Rodinia gpu', 'Datastructure optimized rodinia gpu', 'Direction optimized gpu', 'Datastructure & Direction optimized gpu',
                'GAP (cpu)', 'Rodinia cpu', 'Datastructure optimized rodinia cpu', 'Direction optimized cpu', 'Datastructure & Direction optimized cpu'])
plt.xlabel("Platform and implementation")
plt.ylabel("Execution time (ms)")
plt.title("Execution time for unoptimized Rodinia\n compared to Gunrock and GAP on" + graph)
plt.savefig("exec_" + graph + ".pdf")
#plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.bar([0, 1], 
    [100, gunrock_avg / gpu_avg * 100, gunrock_avg / gpu_dat_avg * 100, gunrock_avg / gpu_dir_avg * 100, gunrock_avg / gpu_dat_dir_avg * 100],
    tick_label=['Gunrock (gpu)', 'Rodinia gpu', 'Datastruct optimized', 'Direction optimized', 'Datastruct & Direction'])
ax1.set_ylabel("Application Efficiency (%)")

ax2.bar([0, 1], 
    [100, gapbs_avg / cpu_avg * 100, gapbs_avg / cpu_dat_avg * 100, gapbs_avg / cpu_dir_avg * 100, gapbs_avg / cpu_dat_dir_avg * 100],
    tick_label=['GAP (cpu)', 'Rodinia cpu', 'Datastructure optimized', 'Direction optimized', 'Datastruct & Direction'])
fig.suptitle("Application Efficiency for unoptimized Rodinia\n compared to Gunrock and GAP on " + graph)
plt.savefig("appeff_" + graph + ".pdf")
#plt.show()