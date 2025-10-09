import os
import fitz
import treeStructure_within_toc
import treeStructure_without_toc

def process_pdf(pdf_path):
    """Process a single PDF file."""
    print(f"\nProcessing: {pdf_path}")
    doc = fitz.open(pdf_path)

    # Check if the PDF has a table of contents (outline)
    if doc.get_toc(simple=True):
        print('PDF has an outline.')
        treeStructure_within_toc.extract_tree(pdf_path)
    else:
        print('PDF has no outline.')
        treeStructure_without_toc.extract_tree(pdf_path)

def main(folder_path):
    """Process all PDF files within a given folder."""
    for filename in os.listdir(folder_path):
        # Only process files ending with .pdf (case insensitive)
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            process_pdf(pdf_path)

if __name__ == "__main__":
    folder_path = "/home/thuyn/pageIndex/PageIndex-Umbrella/tests/pdfs"
    main(folder_path)
