import csv

INPUT_FILE = "traindata/word_ipa.csv"
OUTPUT_FILE = "traindata/word_ipa_no_duplicate.csv"

seen = set()

with open(INPUT_FILE, "r", encoding="utf-8") as inp, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as out:

    reader = csv.reader(inp)
    writer = csv.writer(out)

    header = next(reader)
    writer.writerow(header)

    for word, ipa in reader:
        if word.strip().lower() in seen:
            continue

        seen.add(word.strip().lower())
        writer.writerow([word, ipa])