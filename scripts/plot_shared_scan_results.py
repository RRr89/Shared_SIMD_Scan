import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print('Please pass the csv file from prepare_shared_scan_results.py to this script.')
    sys.exit()

df = pd.read_csv(sys.argv[1], comment='#')

df['runtime_per_predicate'] = df['avg_runtime_ms'] / df['predicate_count']

variants = [v for v in df['variant'].unique() if v.startswith('sse')]
for v in variants:
    df1 = df[df['variant'] == v]
    plt.plot(df1['predicate_count'], df1['runtime_per_predicate'], label=v)

plt.xlabel('# predicates')
plt.ylabel('ms / predicate')
plt.legend()