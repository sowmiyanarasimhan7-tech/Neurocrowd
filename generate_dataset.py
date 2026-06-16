import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_dataset(num_samples=1000):
    data = []

    for _ in range(num_samples):
        situation = np.random.choice(['safe', 'warning', 'critical'], p=[0.4, 0.35, 0.25])

        if situation == 'safe':
            density          = np.random.uniform(0.0, 0.55)
            movement_speed   = np.random.uniform(0.0, 2.2)
            movement_uniform = np.random.uniform(0.0, 0.45)
            counter_flow     = np.random.uniform(0.0, 0.35)
            compression_rate = np.random.uniform(0.0, 0.45)
            audio_spike      = np.random.uniform(0.0, 0.45)
            scream_frequency = np.random.uniform(0.0, 0.35)
            fear_score       = np.random.uniform(0.0, 0.45)
            label            = 0

        elif situation == 'warning':
            density          = np.random.uniform(0.3, 0.8)
            movement_speed   = np.random.uniform(1.2, 3.3)
            movement_uniform = np.random.uniform(0.2, 0.75)
            counter_flow     = np.random.uniform(0.15, 0.65)
            compression_rate = np.random.uniform(0.2, 0.75)
            audio_spike      = np.random.uniform(0.2, 0.75)
            scream_frequency = np.random.uniform(0.15, 0.65)
            fear_score       = np.random.uniform(0.2, 0.75)
            label            = 1

        else:
            density          = np.random.uniform(0.55, 1.0)
            movement_speed   = np.random.uniform(2.2, 5.0)
            movement_uniform = np.random.uniform(0.45, 1.0)
            counter_flow     = np.random.uniform(0.35, 1.0)
            compression_rate = np.random.uniform(0.45, 1.0)
            audio_spike      = np.random.uniform(0.45, 1.0)
            scream_frequency = np.random.uniform(0.35, 1.0)
            fear_score       = np.random.uniform(0.45, 1.0)
            label            = 2

        # extra noise on top of overlapping ranges
        noise = np.random.normal(0, 0.08, 8)
        density          = float(np.clip(density + noise[0], 0, 1))
        movement_speed   = float(max(0, movement_speed + noise[1] * 5))
        movement_uniform = float(np.clip(movement_uniform + noise[2], 0, 1))
        counter_flow     = float(np.clip(counter_flow + noise[3], 0, 1))
        compression_rate = float(np.clip(compression_rate + noise[4], 0, 1))
        audio_spike      = float(np.clip(audio_spike + noise[5], 0, 1))
        scream_frequency = float(np.clip(scream_frequency + noise[6], 0, 1))
        fear_score       = float(np.clip(fear_score + noise[7], 0, 1))

        data.append([
            density, movement_speed, movement_uniform, counter_flow,
            compression_rate, audio_spike, scream_frequency, fear_score, label
        ])

    df = pd.DataFrame(data, columns=[
        'density', 'movement_speed', 'movement_uniformity', 'counter_flow',
        'compression_rate', 'audio_spike', 'scream_frequency', 'fear_score', 'label'
    ])

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/crowd_dataset.csv', index=False)
    print(f"Dataset created: {len(df)} rows")
    print(df['label'].value_counts())
    print(df.head())

if __name__ == "__main__":
    generate_dataset()