import re
import csv

INPUT_FILE = "traindata/de_ipa_lines_clean2.tsv"
OUTPUT_FILE = "traindata/word_ipa.csv"


LAUTSCHRIFT_REGEX = re.compile(r"\{\{Lautschrift\|([^}]*)\}\}")

def extract_first_lautschrift(text):
    match = LAUTSCHRIFT_REGEX.search(text)
    if not match:
        return None

    value = match.group(1).strip()

    if not value:
        return None

    # Kommentare raus 
    value = re.sub(r"\|spr=\w+", "", value)
    value = re.sub(r"spr=\w+\|", "", value)
    value = re.sub(r"\|", "", value)
    value = re.sub(r"spr=\w+", "", value)
    value = re.sub(r"<!--.*?-->", "", value).strip()

    if not value:
        return None

    return value

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:

        writer = csv.writer(outfile)
        writer.writerow(["word", "ipa"])

        for line in infile:
            line = line.strip()
            if "\t" not in line:
                continue

            word, rest = line.split("\t", 1)

            ipa = extract_first_lautschrift(rest)

            if ipa:
                writer.writerow([word, ipa])

if __name__ == "__main__":
    main()