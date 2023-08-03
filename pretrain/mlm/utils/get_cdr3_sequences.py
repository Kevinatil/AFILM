import os
import subprocess
import re
import csv

input_directory = '../predictions_flat'
output_file = './cdrs_sequences.csv'

# Function to run AbRSA and return the output as a string
def run_abrsa(fasta_file):
    command = ['./AbRSA', '-i', fasta_file, '-c', '-o', 'ab_numbering.txt']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

# Function to extract CDR sequences from AbRSA output
def extract_cdrs(output):
    cdr_regex = re.compile(r'(H|L)_CDR(\d)\s*:\s*([A-Za-z]+)')
    cdrs = cdr_regex.findall(output)
    return {f'{chain}_CDR{number}': sequence for chain, number, sequence in cdrs}

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['Filename', 'H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(input_directory):
        if filename.endswith('.fasta'):
            fasta_file = os.path.join(input_directory, filename)
            abrsa_output = run_abrsa(fasta_file)
            cdrs = extract_cdrs(abrsa_output)

            # Append the extracted CDR sequences along with the filename to the CSV file
            cdrs['Filename'] = filename
            writer.writerow(cdrs)


