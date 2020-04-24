import json

import matplotlib.pyplot as plt

if __name__ == '__main__':
    benchmark_data = {int(k): v for k, v in json.load(open('particles_benchmark.json', 'r')).items()}

    particles = sorted(benchmark_data.keys())
    hits = [benchmark_data[n]['hr'] for n in particles]
    times = [benchmark_data[n]['time'] for n in particles]

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    plt.xscale('log')
    ax1.set_xlabel('Number of particles')
    ax1.set_ylabel('HR@10', color=color)
    ax1.plot(particles, hits, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Time (s)', color=color)
    ax2.plot(particles, times, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid()

    fig.tight_layout()

    plt.savefig('pf_experiment.pdf', bbox_inches='tight')
    plt.show()
