import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = r"C:\Users\Sergio\PycharmProjects\rnn_temp_forcast"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname, encoding="utf-8")
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# Headers
# ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"',
# '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"',
# '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
# Data lenght 420,451
# %%

float_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]

# Visualization of the lectures
plt.plot(range(len(temp)), temp)
plt.ylabel("Temp in °C")
plt.xlabel("No. of lectures")
plt.show()

# Visualization for 10 day gap
plt.plot(range(1440), temp[:1440])
plt.ylabel("Temp in °C")
plt.xlabel("No. of lectures")
plt.show()

# %%
# Prepering data for predictions by normalizing the data.
# Will be using the first 200,000 timesteps as training data

mean = temp[:200000].mean(axis=0)
float_data -= mean
std = temp[:200000].std(axis=0)
float_data /= std


# %%
# Creating the generator that yields batches of data from de recent past
# The following parameters will be considered:
# data: The original normalized data
# lookback: How many timesteps back the input data should go
# delay: How many timesteps in the future the target should be
# min and max indexes: delimit which timesteps to draw from
# shuffle: Whether to shuffle the samples or use them in chronological order
# batch_size: number of samples per batch
# step: the period, in timesteps, at which you sample data

def generator(data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = max_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# %%
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=200001,
                      max_index=300000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

tes_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=300001,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)


