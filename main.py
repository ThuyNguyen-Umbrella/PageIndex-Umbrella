import argparse
import os
import json

from pageindex import *
from pageindex.page_index_md import md_to_tree

def main(pdf_path):
    if pdf_path:
        # Validate PDF file
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
        # Process PDF file
        # Configure options
        opt = config(
            model= "Qwen/Qwen3-8B",
            toc_check_page_num= 0,
            max_page_num_each_node= 5,
            max_token_num_each_node= 20000,
            if_add_node_id= "yes",
            if_add_node_summary= "no",
            if_add_doc_description= "no",
            if_add_node_text= "yes",
        )

        # Process the PDF
        toc_with_page_number = page_index_main(pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and index PDF structure.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    args = parser.parse_args()

    main(args.pdf_path)