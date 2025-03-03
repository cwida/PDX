import re
import csv

# Define a function to parse the raw file and extract the required data
def parse_vtune_file(input_file, output_file):
    # Open the input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as csvfile:
        # Initialize the CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=[
            'type', 'n', 'd', 'CPU TIME', 'avg runtime', 'Memory Bound', 'L1 Bound', 'L2 Bound', 'L3 Bound', 'DRAM Bound',
            'Store Bound', 'Loads', 'Stores', 'LLC Misses', 'Average Latency'
        ])
        writer.writeheader()

        # Read the input file line by line
        content = infile.read()

        # Use regex to find each entry starting with "Kernel GATHER"
        type = ['GATHER', 'PDX', 'SIMD']
        entries = re.split(r'Kernel .* \| N=', content)

        for i, entry in enumerate(entries[1:]):  # Skip the first split as it contains no data
            # Initialize a dictionary for the current entry
            data = {}

            # Extract the ID
            id_match = re.search(r'^.*(\d+) D=\d+', entry)
            if id_match:
                n, d = id_match.group(0).split(' ')
                n = n.replace('N=', '')
                d = d.replace('D=', '')
                data['type'] = type[i % 3]
                data['n'] = n
                data['d'] = d

            # Extract CPU TIME
            cpu_time_match = re.search(r'CPU Time:\s+(\d+\.\d+)s', entry)
            runtime = re.search(r'avg:\s+(\d+.\d+)(e-\d+)*', entry)
            data['avg runtime'] = runtime.group(0).split(': ')[1]
            data['CPU TIME'] = cpu_time_match.group(1) if cpu_time_match else ''

            # Extract Memory Bound, L1 Bound, L2 Bound, L3 Bound, DRAM Bound, Store Bound
            metrics = ['Memory Bound', 'L1 Bound', 'L2 Bound', 'L3 Bound', 'DRAM Bound', 'Store Bound']
            for metric in metrics:
                match = re.search(rf'{metric}:\s+(\d+\.\d+)%', entry)
                data[metric] = match.group(1) if match else ''

            # Extract Loads, Stores, LLC Misses, and Average Latency
            loads_match = re.search(r'Loads:\s+([\d,]+)', entry)
            stores_match = re.search(r'Stores:\s+([\d,]+)', entry)
            llc_misses_match = re.search(r'LLC Miss Count:\s+([\d,]+)', entry)
            avg_latency_match = re.search(r'Average Latency \(cycles\):\s+(\d+)', entry)

            data['Loads'] = loads_match.group(1).replace(',', '') if loads_match else ''
            data['Stores'] = stores_match.group(1).replace(',', '') if stores_match else ''
            data['LLC Misses'] = llc_misses_match.group(1).replace(',', '') if llc_misses_match else ''
            data['Average Latency'] = avg_latency_match.group(1) if avg_latency_match else ''

            # Write the extracted data to the CSV file
            writer.writerow(data)

if __name__ == "__main__":
    # Input and output file paths
    input_file = 'raw_metrics.txt'
    output_file = 'parsed_metrics.csv'

    # Run the parser
    parse_vtune_file(input_file, output_file)

    print(f"Data extracted and saved to {output_file}")