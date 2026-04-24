import mwxml
import re
from tqdm import tqdm

INPUT_FILE = "traindata/dewiktionary-latest-pages-articles.xml"
OUTPUT_FILE = "traindata/de_ipa_lines_clean2.tsv"

def is_german(text):
    return ("{{Sprache|Deutsch}}" in text) or ("== Deutsch ==" in text)

def extract_ipa_lines(text):
    lines = text.split("\n")
    ipa_lines = []

    for line in lines:
        line = line.strip()

        if "{{IPA" in line:
            ipa_lines.append(line)

    return ipa_lines

def main():
    dump = mwxml.Dump.from_file(open(INPUT_FILE, "rb"))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for page in tqdm(dump):
            if page.namespace != 0:
                continue

            title = page.title

            for revision in page:
                text = revision.text
                if not text:
                    continue

                if not is_german(text):
                    continue

                ipa_lines = extract_ipa_lines(text)

                for ipa in ipa_lines:
                    out.write(f"{title}\t{ipa}\n")

                break

if __name__ == "__main__":
    main()