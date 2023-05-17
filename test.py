from math import log2
for signal_size in [7, 14, 28, 56, 112, 224, 401]:
    num_steps = int(log2(signal_size / 4))
    print(f'Curr sig size: {signal_size} \n #### num steps: {num_steps}')