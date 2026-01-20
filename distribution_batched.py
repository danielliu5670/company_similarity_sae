import numpy as np
import plotly.graph_objects as go
from datasets import load_dataset
import argparse


P = argparse.ArgumentParser()
P.add_argument("--img_path", type=str, default="images/distribution_summed_sae_features.png", help="Path to save the image")
P.add_argument("--batch_size", type=int, default=100, help="Number of samples to process per batch")
args = P.parse_args()

# Configuration
DESIRED_LENGTH = 131072
NUM_BINS = 300
VALUE_RANGE = (0, 7.5)
CLIP_THRESHOLD = 7.5

# Initialize histogram accumulator
hist_accumulator = np.zeros(NUM_BINS, dtype=np.int64)
total_values = 0
values_above_threshold = 0

# Load dataset in streaming mode to avoid loading everything into memory
ds = load_dataset("marco-molinari/company_reports_with_features", split="train")

# Process in batches
num_samples = len(ds)
num_batches = (num_samples + args.batch_size - 1) // args.batch_size

print(f"Processing {num_samples} samples in {num_batches} batches...")

for batch_idx in range(num_batches):
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, num_samples)
    
    # Select batch
    batch = ds.select(range(start_idx, end_idx))
    
    # Process each sample in the batch
    batch_values = []
    for sample in batch:
        arr = np.array(sample['features'])
        if len(arr.shape) > 0 and arr.shape[0] == DESIRED_LENGTH:
            batch_values.extend(arr)
    
    if batch_values:
        batch_array = np.array(batch_values)
        
        # Count values above threshold before clipping
        values_above_threshold += np.sum(batch_array > CLIP_THRESHOLD)
        total_values += len(batch_array)
        
        # Clip values
        clipped_batch = np.clip(batch_array, None, CLIP_THRESHOLD)
        
        # Compute histogram for this batch and accumulate
        batch_hist, _ = np.histogram(clipped_batch, bins=NUM_BINS, range=VALUE_RANGE)
        hist_accumulator += batch_hist
        
        # Free memory
        del batch_values, batch_array, clipped_batch, batch_hist
    
    if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
        print(f"  Processed batch {batch_idx + 1}/{num_batches}")

print(f"Total values processed: {total_values:,}")

# Calculate final statistics
pct_clipped = 100 * (values_above_threshold / total_values) if total_values > 0 else 0

# Calculate bin centers and proportions
bin_edges = np.linspace(VALUE_RANGE[0], VALUE_RANGE[1], NUM_BINS + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
proportions = hist_accumulator / hist_accumulator.sum()

# Create plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=bin_centers,
    y=proportions,
    marker_color='rgb(0,68,136)',
))

# Style the plot
fig.update_layout(
    xaxis=dict(
        title="Feature Value",
        title_font=dict(size=48),
        tickfont=dict(size=40),
        gridcolor="rgba(0, 0, 0, 0.2)",
        title_standoff=44,
        showline=False,
        zeroline=False
    ),
    yaxis=dict(
        title="Proportion",
        title_font=dict(size=48),
        tickfont=dict(size=40),
        tickformat=".2f",
        showgrid=False,
        title_standoff=80,
        showline=False,
        side='left',
        automargin=True,
        ticksuffix="   ",
        ticklabelposition="outside right",
    ),
    template="plotly_white",
    margin=dict(t=50, b=110, l=200, r=165),
    width=1430,
    height=990,
    plot_bgcolor='white'
)

fig.update_yaxes(range=(0, 0.5))
fig.update_xaxes(range=(0, 7.5), constrain='domain')

if pct_clipped > 0:
    fig.add_annotation(
        x=bin_centers[-1],
        y=proportions[-1] * 1.12,
        text=f"Proportion > 7.5: {(pct_clipped / 100):.2f}",
        showarrow=False,
        font=dict(size=30, color="rgb(0,68,136)"),
        yshift=25
    )

fig.write_image(args.img_path, scale=1, width=1920, height=1080)
print(f"Image saved to {args.img_path}")
