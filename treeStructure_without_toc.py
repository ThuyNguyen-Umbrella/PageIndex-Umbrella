import argparse
import os
import json
import fitz
import uuid
import re
import unicodedata

from pageindex import *
from pageindex.page_index_md import md_to_tree


def generate_node_id():
    return str(uuid.uuid4())[:8]

def get_outline(pdf_path):
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
            if_add_node_text= "no",
        )

        # Process the PDF
        toc_with_page_number = page_index_main(pdf_path, opt)
        # toc_with_page_number = json.loads(toc_with_page_number)
        print('Parsing done')
        
    return toc_with_page_number

def get_text_from_range(doc, start, end):
    text = ""
    for p in range(start - 1, end):
        text += doc.load_page(p).get_text() + "\n"
    return text.strip()

def build_tree_with_toc(doc, outline):
    nodes = []
    stack = []

    full_text_by_page = [doc.load_page(p).get_text() for p in range(doc.page_count)]

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def make_pattern(title: str):
        clean = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title).strip()

        escaped = re.escape(clean)
        flexible = re.sub(r"\\ ", r"\\s+", escaped)
        pattern_str = r"(?:\d+(?:\.\d+)*\.\s*)?" + flexible

        # return re.compile(
        #     r"(?:\d+(?:\.\d+)*\.\s*)?" + re.escape(clean),
        #     flags=re.MULTILINE
        # )
        return re.compile(pattern_str, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    def get_level(structure_str):
        return len(structure_str.split("."))
    
    def get_text_by_title(start_page, end_page, title, next_title=None):
        text = "".join(full_text_by_page[start_page - 1:end_page])
        norm_text = normalize(text)
        # print("text:", norm_text)
        norm_title = normalize(title)
        # print("title:", norm_title)
        norm_next = normalize(next_title) if next_title else None
        # print("next_title:", norm_next)

        pattern_title = make_pattern(norm_title)
        match_title = pattern_title.search(norm_text)

        if not match_title:
            return norm_text.strip()
        
        start_pos = match_title.start()

        if norm_next:

            pattern_next = make_pattern(norm_next)
            match_next = pattern_next.search(norm_text, match_title.end())
            if match_next:
                return norm_text[start_pos:match_next.start()].strip()


        return norm_text[start_pos:].strip()
    
    for i, value in enumerate(outline):
        print('type of value', type(value))
        # print('value:', value)

        if isinstance(value, str):
            value = json.loads(value)
            print('type:', type(value))
            print('value', value)

        if i < len(outline) -1:
            end_idx = outline[i + 1]['physical_index']
            next_title = outline[i + 1]['title']
        else:
            end_idx = doc.page_count
            next_title = None
        
        text = get_text_by_title(value['physical_index'], end_idx, value['title'], next_title)

        node =  {
            "title": value['title'],
            "node_id": generate_node_id(),
            "start_index": value['physical_index'],
            # "summary": res,
            "text": text,
            "nodes": []
        }

        level = get_level(value["structure"])

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            nodes.append(node)

        stack.append((level, node))

    return nodes

def extract_tree(pdf_path):
    doc = fitz.open(pdf_path)
    # metadata = doc.metadata
    outline = get_outline(pdf_path)

    if not outline:
        print("can't create outline.")
        # tree = build_tree_without_toc(pdf_path)
        # return tree
    else:
        print("outline was created")
        tree = build_tree_with_toc(doc, outline)

    # Save results
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]    
        output_dir = './tests/results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')
    doc.close()




    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract and index PDF structure.")
    # parser.add_argument("pdf_path", help="Path to the PDF file.")
    # args = parser.parse_args()

    # main(args.pdf_path)
    pdf_path = '/home/thuyn/pageIndex/PageIndex-Umbrella/tests/pdfs/P4.pdf'
    extract_tree(pdf_path)
    # print(get_outline(pdf_path))
    # outline_path = '/home/thuyn/pageIndex/PageIndex-Umbrella/tests/results/phishing_structure.json'
    # with open(outline_path, "r", encoding="utf-8") as f:
    #     outline = json.load(f)
    # print(type(outline))
    # print(type(outline[0]))
    # doc = fitz.open(pdf_path)
    # tree = build_tree_with_toc(doc, outline)
    # with open('tree.json', 'w', encoding='utf-8') as f:
    #     json.dump(tree, f, indent=2)