#!/usr/bin/env python3
import re
import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_gprof(filename):
    """
    Parse the "Flat profile" section of a gprof output file
    and return a dictionary mapping function names to their self time.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Locate the "Flat profile:" section.
    start = None
    for i, line in enumerate(lines):
        if "Flat profile:" in line:
            start = i
            break
    if start is None:
        print(f"Flat profile section not found in {filename}")
        return {}

    # Next, locate the header line that contains the column titles.
    header = None
    for j in range(start, len(lines)):
        if re.search(r"\s*time\s+seconds\s+seconds", lines[j]):
            header = j
            break
    if header is None:
        print(f"Profile header not found in {filename}")
        return {}

    profile_data = {}
    # The data typically starts from the line after header.
    for line in lines[header+1:]:
        if not line.strip():
            # Break on an empty line (sometimes marks the end of the table)
            break
        # Example line format:
        # " 25.00     5.00     5.00  100     0.05     0.10  compute"
        # We'll extract the "self" seconds (3rd column) and the function name (last column).
        m = re.match(r"\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+).*?\s+([^\s].*)", line)
        if m:
            percent = float(m.group(1))
            cumulative = float(m.group(2))
            self_time = float(m.group(3))
            func_name = m.group(4).strip()
            # Store self time per function.
            profile_data[func_name] = self_time
    return profile_data

def consolidate_profiles(profile_files):
    """
    Given a list of profile file names, consolidate the data into a dictionary:
    { rank: { func: self_time, ... }, ... }
    It assumes profile filenames contain the rank number at the end, e.g., profile_rank0.txt.
    """
    consolidated = {}
    for fname in profile_files:
        # Try to extract the rank. For example, if filename is profile_rank0.txt, capture "0".
        m = re.search(r'rank(\d+)', fname)
        if m:
            rank = int(m.group(1))
        else:
            # If no rank found, use filename as key.
            rank = fname
        data = parse_gprof(fname)
        consolidated[rank] = data
    return consolidated

def visualize_profiles(consolidated):
    """
    Visualize consolidated gprof flat profile data across ranks.
    This function creates a bar chart comparing the self times for key functions across MPI ranks.
    """
    # Identify the union set of function names across all ranks.
    all_funcs = set()
    for rank, data in consolidated.items():
        all_funcs.update(data.keys())
    # For clarity, you might choose to only visualize the top N functions 
    # (for example, those with the highest total self time across all ranks).
    all_funcs = list(all_funcs)
    
    # Build a matrix where rows correspond to functions and columns to MPI ranks.
    ranks = sorted(consolidated.keys())
    func_list = sorted(all_funcs)
    data_matrix = []
    for func in func_list:
        row = []
        for rank in ranks:
            time_val = consolidated[rank].get(func, 0)
            row.append(time_val)
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    
    # Plot the data as grouped bar charts.
    x = np.arange(len(func_list))  # number of functions
    width = 0.8 / len(ranks)  # width of each bar
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, rank in enumerate(ranks):
        ax.bar(x + i * width, data_matrix[:, i], width, label=f'Rank {rank}')
    
    ax.set_xlabel('Function')
    ax.set_ylabel('Self Time (seconds)')
    ax.set_title('Consolidated gprof Flat Profile: Self Time per Function across MPI ranks')
    ax.set_xticks(x + width * (len(ranks) - 1) / 2)
    ax.set_xticklabels(func_list, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Find all profile files matching a pattern. Modify the glob pattern as needed.
    # For example, if you have files like profile_rank0.txt, profile_rank1.txt, ...
    profile_files = glob.glob("profile_rank*.txt")
    if not profile_files:
        print("No profile files found matching 'profile_rank*.txt'.")
        sys.exit(1)
    
    consolidated = consolidate_profiles(profile_files)
    visualize_profiles(consolidated)
