import matplotlib.pyplot as plt

# Read the analysis.txt file
with open('analysis.txt', 'r') as f:
    lines = f.readlines()

# Flag to indicate when the flat profile starts
is_flat_profile = False

# Lists to hold the function names and their corresponding times
function_names = []
self_times = []

# Parse the file to extract the relevant data
for line in lines:
    if line.startswith("Flat profile:"):
        is_flat_profile = True
        continue

    if is_flat_profile and line.startswith("  % time"):
        continue  # Skip the header line

    if is_flat_profile and len(line.split()) > 5:
        # Extract the function name and self time
        parts = line.split()
        percent_time = parts[0]
        self_time = parts[1]
        function_name = " ".join(parts[5:])

        # Check if self_time is a valid float
        try:
            self_time = float(self_time)
            function_names.append(function_name)
            self_times.append(self_time)
        except ValueError:
            # Ignore lines where self_time is not a valid number
            continue

# Plotting the results
plt.figure(figsize=(10, 6))
plt.barh(function_names, self_times, color='skyblue')
plt.xlabel('Time in Seconds')
plt.title('Function Execution Time (Self Time) from gprof Profile')
plt.tight_layout()
plt.show()
