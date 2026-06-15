import pandas as pd
import numpy as np
import os

# Set random seed so we get same data every time we run
np.random.seed(42)

def generate_dataset(num_samples=1000):
    data = []

    for _ in range(num_samples):
        # Randomly pick a situation type
        situation = np.random.choice(['safe', 'warning', 'critical'], p=[0.4, 0.35, 0.25])

        if situation == 'safe':
            density          = np.random.uniform(0.0, 0.4)
            movement_speed   = np.random.uniform(0.0, 1.5)
            movement_uniform = np.random.uniform(0.0, 0.3)
            counter_flow     = np.random.uniform(0.0, 0.2)
            compression_rate = np.random.uniform(0.0, 0.3)
            audio_spike      = np.random.uniform(0.0, 0.3)
            scream_frequency = np.random.uniform(0.0, 0.2)
            fear_score       = np.random.uniform(0.0, 0.3)
            label            = 0  # Safe

        elif situation == 'warning':
            density          = np.random.uniform(0.4, 0.7)
            movement_speed   = np.random.uniform(1.5, 2.8)
            movement_uniform = np.random.uniform(0.3, 0.6)
            counter_flow     = np.random.uniform(0.2, 0.5)
            compression_rate = np.random.uniform(0.3, 0.6)
            audio_spike      = np.random.uniform(0.3, 0.6)
            scream_frequency = np.random.uniform(0.2, 0.5)
            fear_score       = np.random.uniform(0.3, 0.6)
            label            = 1  # Warning

        else:  # critical
            density          = np.random.uniform(0.7, 1.0)
            movement_speed   = np.random.uniform(2.8, 5.0)
            movement_uniform = np.random.uniform(0.6, 1.0)
            counter_flow     = np.random.uniform(0.5, 1.0)
            compression_rate = np.random.uniform(0.6, 1.0)
            audio_spike      = np.random.uniform(0.6, 1.0)
            scream_frequency = np.random.uniform(0.5, 1.0)
            fear_score       = np.random.uniform(0.6, 1.0)
            label            = 2  # Critical

        data.append([
            density,
            movement_speed,
            movement_uniform,
            counter_flow,
            compression_rate,
            audio_spike,
            scream_frequency,
            fear_score,
            label
        ])

    # Create dataframe
    df = pd.DataFrame(data, columns=[
        'density',
        'movement_speed',
        'movement_uniformity',
        'counter_flow',
        'compression_rate',
        'audio_spike',
        'scream_frequency',
        'fear_score',
        'label'
    ])

    # Save to data folder
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/crowd_dataset.csv', index=False)
    print(f"Dataset created: {len(df)} rows")
    print(df['label'].value_counts())
    print(df.head())

if __name__ == "__main__":
    generate_dataset()