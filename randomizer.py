import sys
import csv
import pdb
import random

error_rate = float(sys.argv[2]) / 100.0 
print(error_rate)

output = []
pdb.set_trace()
errors_inserted = 0
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    output.append(reader.fieldnames)
    for line in reader:
        line_out = []
        for val in line.values():
            toss = random.random()
            if toss > error_rate:
                line_out.append(float(val))
            else:
                errors_inserted = errors_inserted + 1
                line_out.append("")
        line_out[-1] = int(line_out[-1]) if line_out[-1] else ""
        output.append(line_out)

print(f"Errors inserted {errors_inserted}")
print()
print(",".join(output[0]))
for out in output[1:]:
    print(",".join(map(str,out)))

