import numpy as np
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("marco-molinari/company_reports_with_features", split="train")

print(f"Dataset size: {len(ds)} samples\n")

# Inspect the first few samples
print("=" * 60)
print("INSPECTING FIRST 3 SAMPLES")
print("=" * 60)

for i in range(min(3, len(ds))):
    sample = ds[i]
    features = sample['features']
    
    print(f"\nSample {i}:")
    print(f"  Type of 'features': {type(features)}")
    
    if isinstance(features, list):
        print(f"  Length of features list: {len(features)}")
        if len(features) > 0:
            print(f"  Type of first element: {type(features[0])}")
            if isinstance(features[0], list):
                print(f"  Length of first sub-list: {len(features[0])}")
            elif hasattr(features[0], 'shape'):
                print(f"  Shape of first element: {features[0].shape}")
    
    # Try converting to numpy array
    arr = np.array(features)
    print(f"  After np.array(): shape={arr.shape}, dtype={arr.dtype}")
    
    # Check if it's nested
    if arr.dtype == object:
        print("  WARNING: Object dtype detected - likely nested arrays")
        if len(arr) > 0:
            inner = np.array(arr[0])
            print(f"  First inner array: shape={inner.shape}, dtype={inner.dtype}")

print("\n" + "=" * 60)
print("COMPARING WITH PANDAS APPROACH (original code)")
print("=" * 60)

# Load small subset as pandas to compare
small_ds = ds.select(range(min(5, len(ds))))
df = small_ds.to_pandas()

print(f"\nPandas DataFrame columns: {df.columns.tolist()}")
print(f"Type of df['features']: {type(df['features'])}")
print(f"Type of df['features'].values: {type(df['features'].values)}")

if len(df) > 0:
    first_feat = df['features'].iloc[0]
    print(f"Type of first features entry: {type(first_feat)}")
    first_arr = np.array(first_feat)
    print(f"First features as array: shape={first_arr.shape}, dtype={first_arr.dtype}")

print("\n" + "=" * 60)
print("CHECKING LENGTH FILTER (DESIRED_LENGTH = 131072)")
print("=" * 60)

DESIRED_LENGTH = 131072
matching_count = 0
length_distribution = {}

for i in range(min(100, len(ds))):
    sample = ds[i]
    features = sample['features']
    arr = np.array(features)
    
    # Handle potential nesting
    if arr.dtype == object and len(arr) > 0:
        # Try flattening or accessing inner arrays
        try:
            arr = np.concatenate([np.array(x) for x in arr])
        except:
            arr = np.array(arr[0]) if len(arr) == 1 else arr
    
    shape_key = str(arr.shape)
    length_distribution[shape_key] = length_distribution.get(shape_key, 0) + 1
    
    if len(arr.shape) > 0 and arr.shape[0] == DESIRED_LENGTH:
        matching_count += 1

print(f"\nOut of first {min(100, len(ds))} samples:")
print(f"  Samples matching length {DESIRED_LENGTH}: {matching_count}")
print(f"\nShape distribution:")
for shape, count in sorted(length_distribution.items(), key=lambda x: -x[1]):
    print(f"  {shape}: {count} samples")

print("\n" + "=" * 60)
print("VALUE RANGE CHECK")
print("=" * 60)

# Check actual value ranges
all_mins = []
all_maxs = []
for i in range(min(10, len(ds))):
    sample = ds[i]
    arr = np.array(sample['features'])
    if arr.dtype == object and len(arr) > 0:
        try:
            arr = np.concatenate([np.array(x) for x in arr])
        except:
            pass
    if arr.dtype != object and arr.size > 0:
        all_mins.append(float(np.min(arr)))
        all_maxs.append(float(np.max(arr)))

if all_mins:
    print(f"Min values across samples: {min(all_mins):.4f} to {max(all_mins):.4f}")
    print(f"Max values across samples: {min(all_maxs):.4f} to {max(all_maxs):.4f}")
    print(f"Histogram range is (0, 7.5) - values outside this won't appear")
else:
    print("Could not extract numeric values from features")
