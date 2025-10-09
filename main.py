import fitz
import treeStructure_within_toc 
import treeStructure_without_toc

def main(pdf_path):
    doc = fitz.open(pdf_path)
    
    if doc.get_toc(simple=True):
        print('PDF has outline.')
        treeStructure_within_toc.extract_tree(pdf_path)
    else:
        print('PDF has no outline.')
        treeStructure_without_toc.extract_tree(pdf_path)
    
if __name__ == "__main__":
    pdf_path = '/home/thuyn/pageIndex/PageIndex-Umbrella/tests/pdfs/TR-03121-1_Biometrics_7_0_draft2.pdf'
    main(pdf_path)
        
        
