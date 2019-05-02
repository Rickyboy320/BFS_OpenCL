import sys

if(len(sys.argv) < 3):
    print("Not enough arguments supplied. Needs 2 (from, to).")
    exit()

file_from = sys.argv[1]
file_to = sys.argv[2]

print("Converting", file_from, "to", file_to)

is_integer = False
is_real = False
with open(file_from, 'r') as temp:
    line = temp.readline()
    if len(line.split(" ")) == 3:
        try:
            int(line.split(" ")[2])
            is_integer = True
        except ValueError:
            is_real = True

banner = "%%MatrixMarket matrix coordinate "
if is_integer:
    banner += "Integer"
elif is_real:
    banner += "real"
else:
    banner += "pattern"

banner += " symmetric\n"

maxval = 0
count = 0
zero_based = False
with open(file_from, 'r') as original: 
    for line in original:
        splitted = line.split(" ")
        count += 1
        if int(splitted[0]) == 0 or int(splitted[1]) == 0:
            zero_based = True
        if int(splitted[0]) > maxval:
            maxval = int(splitted[0])
        if int(splitted[1]) > maxval:
            maxval = int(splitted[1])


if not zero_based:
    sizes = str(maxval + 1) + " " + str(maxval + 1) + " " + str(count) + "\n"

    with open(file_from, 'r') as original: data = original.read()
    with open(file_to, 'w') as modified: modified.write(banner + sizes + data)
else:
    sizes = str(maxval + 2) + " " + str(maxval + 2) + " " + str(count) + "\n"

    with open(file_from, 'r') as original: 
        with open(file_to, 'w') as modified: 
            modified.write(banner + sizes)

            for line in original:
                splitted = line.split(" ")
                modified.write(str(int(splitted[0]) + 1) + " " + str(int(splitted[1]) + 1))
                if is_integer or is_real:
                    modified.write(splitted[1]) 
                modified.write("\n") 


