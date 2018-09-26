import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print('Please pass the csv file from prepare_shared_scan_results.py to this script.')
    sys.exit()

df = pd.read_csv(sys.argv[1], comment='#')

df['runtime_per_predicate'] = df['avg_runtime_ms'] / df['predicate_count']

# figure 1: absolute
plt.figure()
variants = [v for v in df['variant'].unique() if v.startswith('sse')]
for v in variants:
    df1 = df[df['variant'] == v]
    plt.plot(df1['predicate_count'], df1['avg_runtime_ms'], label=v)

plt.title('Shared Scan performance (absolute)')
plt.xlabel('# predicates')
plt.ylabel('ms')
plt.legend()

# figure 2: relative to predicate count
plt.figure()
variants = [v for v in df['variant'].unique() if v.startswith('sse')]
for v in variants:
    df1 = df[df['variant'] == v]
    plt.plot(df1['predicate_count'], df1['runtime_per_predicate'], label=v)

plt.title('Shared Scan performance (relative to predicate count)')
plt.xlabel('# predicates')
plt.ylabel('ms / predicate')
plt.legend()

# figure 3: improvement to sequential
plt.figure()

g = df.groupby(df['predicate_count'])
rows = []
for name, group in g:
    r1 = group[group['variant'] == 'sse 128, sequential (unrolled)']['avg_runtime_ms'].values[0]
    r2 = group[group['variant'] == 'sse 128, standard']['avg_runtime_ms'].values[0]
    rows.append({'predicate_count': name, 'improvement': 1 - r2 / r1})
df3 = pd.DataFrame(rows)

plt.bar(df3['predicate_count'], df3['improvement'], width=1)
plt.title('Improvement standard vs sequential (unrolled)')
plt.xlabel('# predicates')
plt.ylabel('improvement')
plt.gca().set_ylim(-0.5, 0.5)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))