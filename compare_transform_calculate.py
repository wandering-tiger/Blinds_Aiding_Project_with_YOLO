import pandas as pd

# Load the CSV file
CSV_INPUT_PATH = "speed_comparison.csv"
CSV_OUTPUT_PATH = "filtered_speed_comparison.csv"

df = pd.read_csv(CSV_INPUT_PATH)

# Define speed threshold for large speed difference
max_speed_diff = 50

# Define speed threshold (for example, a maximum valid speed, adjust as needed)
max_speed = 100  # Set maximum speed threshold (in meters per second)
min_speed = 0.1  # Set minimum speed threshold (to avoid zero or negative speeds)

# Filter out rows where Speed_1 or Speed_2 exceed the valid speed range
df = df[(df["Speed_1 (m/s)"] <= max_speed) & (df["Speed_1 (m/s)"] >= min_speed)]
df = df[(df["Speed_2 (m/s)"] <= max_speed) & (df["Speed_2 (m/s)"] >= min_speed)]

# Remove extreme outliers using IQR for Speed_1
Q1 = df["Speed_1 (m/s)"].quantile(0.25)
Q3 = df["Speed_1 (m/s)"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df["Speed_1 (m/s)"] <= upper_bound]

# Remove extreme outliers using IQR for Speed_2
Q1 = df["Speed_2 (m/s)"].quantile(0.25)
Q3 = df["Speed_2 (m/s)"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df["Speed_2 (m/s)"] <= upper_bound]

# Recalculate speed error
df["Recalculated_Error%"] = (abs(df["Speed_1 (m/s)"] - df["Speed_2 (m/s)"]) / abs(df["Speed_1 (m/s)"])) * 100

# Remove rows where the absolute difference between Speed_1 and Speed_2 exceeds max_speed_diff
df = df[df["Recalculated_Error%"] <= max_speed_diff]

# Save the filtered data
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Filtered speed data saved to {CSV_OUTPUT_PATH}")

# Compute the new average error
mean_error = df["Recalculated_Error%"].mean()
print(f"Average speed error: {mean_error:.2f}%")
