from operator import add

# (roughly) parses a callgrind file and returns the cost for a given function
def get_inclusive_cost(file_name, function_name):
    alias = ''
    collecting = False
    data = []
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n', '')

            # basic settings
            if line.startswith('positions: '):
                positions = line[len('positions: '):].split(' ')
                continue

            if line.startswith('events: '):
                events = line[len('events: '):].split(' ')
                data = len(events) * [0]
                continue

            # grab function alias
            if function_name in line and '=(' in line:
                alias = line[line.index('(') : line.index(')')+1]
                continue

            # start collecting
            if line.startswith('fn=') and (function_name in line or (len(alias) > 0 and alias in line)):
                collecting = True
                continue
            
            # end of collection
            if collecting and line.startswith('fn='):
                return dict(zip(events, data))

            # cost line
            if collecting and not '=' in line:
                cost = line.split(' ')[len(positions):] # skip positions
                cost = list(map(lambda x: int(x), cost)) # convert to int
                cost += (len(events) - len(cost)) * [0] # pad to full length
                data = [sum(x) for x in zip(data, cost)] # add to previous data
                continue

    return dict(zip(events, data))