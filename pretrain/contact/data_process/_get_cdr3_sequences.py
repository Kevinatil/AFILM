import os
import subprocess
import re

input_directory = '../data/IG/raw/oas/predictions_flat'
tool_dir='../tool/AbRSA/AbRSA'

# Function to run AbRSA and return the output as a string
def run_abrsa(fasta_file):
    command = [tool_dir, '-i', fasta_file, '-c', '-o', 'ab_numbering.txt']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

# Function to extract CDR sequences from AbRSA output
def extract_cdrs(output):
    cdr_regex = re.compile(r'(H|L)_CDR(\d)\s*:\s*([A-Za-z]+)')
    cdrs = cdr_regex.findall(output)
    print(cdrs)
    return {f'{chain}_CDR{number}': sequence for chain, number, sequence in cdrs}

all_files=os.listdir(input_directory)[:10]

for filename in all_files:
    if filename.endswith('.fasta'):
        fasta_file = os.path.join(input_directory, filename)
        abrsa_output = run_abrsa(fasta_file)
        cdrs = extract_cdrs(abrsa_output)
        
        