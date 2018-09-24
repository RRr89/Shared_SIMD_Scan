import sys
import os
import subprocess
import csv

if len(sys.argv) < 2:
    print('Please pass the path to the shared simd scan binary to this script.')
    sys.exit()

simdscan = sys.argv[1]
csv_writer = csv.writer(sys.stdout)
csv_writer.writerow(['predicate_count', 'variant', 'avg_runtime_ms'])

def parse_output(predicate_count, output):
    for line in output.splitlines():
        if not line.startswith('*'):
            continue
        variant = line[2:].split(': ')[0]
        avg_runtime_ms = line.split(': ')[1].split('; ')[0][:-2]
        csv_writer.writerow([predicate_count, variant, avg_runtime_ms])

for i in range(1, 33):
    predicate_count = i
    print('# predicate count: {}'.format(predicate_count))
    result = subprocess.run([simdscan, 'sharedscan', str(predicate_count)], stdout=subprocess.PIPE)
    parse_output(predicate_count, result.stdout.decode('utf-8'))