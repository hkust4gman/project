from tqdm import tqdm
import csv
import re


def generate_n_samples(file_path, n):
    """
    Generates n samples from the given file.
    :param file_path: Path to the file.
    :param n: Number of samples to generate.
    :return:
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = ""
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == "":
                samples.append(chunk)
                chunk = ""
            chunk += line
            if len(samples) >= n:
                break
    with open('D:/Data/Amazon Dataset/sample.txt', 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample)
    return samples


def convert_to_csv(input_file_path, output_file_path):
    """
    Converts the input file to a CSV file.
    :param input_file_path: Path to the input file.
    :param output_file_path: Path to the output file.
    :return:
    """
    headers = ["product/productId", "product/title", "product/price", "review/userId", "review/profileName",
               "review/helpfulness", "review/score", "review/time", "review/summary", "review/text"]
    num_lines = 381554470
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_path, 'w', encoding='utf-8', newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers, restval="", )
        writer.writeheader()
        chunk = {}
        progress_bar = tqdm(desc="Processing", unit="lines", total=num_lines)
        while True:
            line = input_file.readline()
            if not line:
                break
            progress_bar.update(1)
            if line.strip() == "":
                # write the chunk to the csv file
                writer.writerow(chunk)
                chunk = {}
                continue
            line_list = re.split(r':\s*', line.strip())
            if len(line_list) >= 2:
                chunk[line_list[0]] = ': '.join(line_list[1:])
            else:
                print("Error: ", line_list)
