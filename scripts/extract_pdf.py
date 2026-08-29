import fitz
from pathlib import Path

# Locate the project root
PROJECT_ROOT = Path(__file__).parent.parent

# Path to the PDF
pdf_path = PROJECT_ROOT / "data" / "catalogs" / "RUBIS 95_85.pdf"

# Folder to save extracted text
output_folder = PROJECT_ROOT / "data" / "extracted"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "rubis95_85.txt"

# Open the PDF
document = fitz.open(pdf_path)

all_text = ""

# Read every page
for page_number, page in enumerate(document):

    print(f"Reading page {page_number + 1}")

    text = page.get_text()

    all_text += f"\n\n========== PAGE {page_number + 1} ==========\n\n"

    all_text += text

# Save everything
with open(output_file, "w", encoding="utf-8") as file:
    file.write(all_text)

print("\nDone!")
print(f"Extracted text saved to:\n{output_file}")