import sys
import os
import subprocess
import csv

if len(sys.argv) < 2:
    print('Please pass the path to the shared simd scan binary to this script.')
    sys.exit()

simdscan = sys.argv[1]
csv_writer = csv.writer(sys.stdout)
csv_writer.writerow(['data_size', 'predicate_count', 'variant', 'avg_runtime_ms'])

def parse_output(data_size, predicate_count, output):
    for line in output.splitlines():
        if not line.startswith('*'):
            continue
        variant = line[2:].split(': ')[0]
        avg_runtime_ms = line.split(': ')[1].split('; ')[0][:-2]
        csv_writer.writerow([data_size, predicate_count, variant, avg_runtime_ms])

def run(data_size, predicate_count, repetitions):
    print('# data size: {}, predicate count: {}'.format(data_size, predicate_count))
    result = subprocess.run([simdscan, str(data_size), str(repetitions), 'sharedscan', str(predicate_count)], stdout=subprocess.PIPE)
    print('# {}'.format(result.args))
    parse_output(data_size, predicate_count, result.stdout.decode('utf-8'))

repetitions = 1
for data_size in [40]:
    for predicate_count in range(1, 513, 1):
        run(data_size, predicate_count, repetitions)