import json
import time


def main():
    with open('data/cpu_util.json', 'rb') as f:
        src = json.load(f)

    title = ['timestamp']
    time_values = {}

    for item in src['data']['result']:
        metric = item['metric']['__name__']
        values = item['values']
        title.append(metric)
        for t, v in values:
            time_values.setdefault(t, [])
            time_values[t].append(v)

    with open('data/cpu_util.csv', 'w+') as f:
        f.writelines(','.join(title) + '\n')
        for t, v in time_values.items():
            d = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
            f.writelines(f'{d},{",".join(v)}\n')


if __name__ == '__main__':
    main()
