# from math import log2
# for signal_size in [7, 14, 28, 56, 112, 224, 401]:
#     num_steps = int(log2(signal_size / 4))
#     print(f'Curr sig size: {signal_size} \n #### num steps: {num_steps}')

import torch
from math import log2

real = torch.randn((48, 1, 401))
desired_steps = 6
signal_sizes = []
scale_factor = []

for steps in range(desired_steps + 1):
    if steps == 0:
        signal_sizes.append(real.shape[2])
    else:
        signal_sizes.append(signal_sizes[steps-1] // 2)

signal_sizes.sort()
print(signal_sizes)


last_signal_size = 0
for signal_size in signal_sizes:
    if not signal_size == signal_sizes[0]:
        scale_factor.append(signal_size / last_signal_size)
    last_signal_size = signal_size

print(scale_factor)
# for sigal_size in signal_sizes:
#     print(f'{sigal_size}: {int(log2(signal_size / 4))}')
for idx, signal_size in enumerate(signal_sizes):
    print(f'sigal_size: {signal_size}, idx: {idx}')

# desired_steps = 6
# factors = [1, 1, 1, 1, 1, 1, 1]
# factors2 = [1 for _ in range(desired_steps + 1)]
# assert len(factors) == len(factors2)
# for i in factors:
#     assert factors[i] == factors2[i]
# print('success')
