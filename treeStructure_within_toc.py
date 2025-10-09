import fitz
import json
import uuid
import re
import unicodedata
import os
import argparse


def generate_node_id():
    return str(uuid.uuid4())[:8]


def get_outline(doc):
    #[level, title, page_number, ...]
    return doc.get_toc(simple=True)


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

    # def make_pattern(title: str):
    #     clean = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title).strip()

    #     return re.compile(
    #         r"(?:\d+(?:\.\d+)*\.\s*)?" + re.escape(clean),
    #         flags=re.MULTILINE
    #     )

    def make_pattern(title: str):
        clean = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title).strip()

        escaped = re.escape(clean)
        flexible = re.sub(r"\\ ", r"\\s+", escaped)
        pattern_str = r"(?:\d+(?:\.\d+)*\.\s*)?" + flexible
        
        return re.compile(pattern_str, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

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

    for i, (level, title, start_idx) in enumerate(outline):

        if i < len(outline) - 1:
            end_idx = outline[i + 1][2]
            next_title = outline[i + 1][1]
        else:
            end_idx = doc.page_count
            next_title = None


        text = get_text_by_title(start_idx, end_idx, title, next_title)

        # prompt = f"""
        # You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.
        # The description must be written in the same language as the partial document text.

        # Partial Document Text: {text}
    
        # Directly return the description, do not include any other text.
        # """
        # model = OpenAIModel()
        # res = model.response(prompt)

        node = {
            "title": title,
            "node_id": generate_node_id(),
            "start_index": start_idx,
            # "summary": res,
            "text": text,
            "nodes": []
        }

        # print("title:", title)

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
    outline = get_outline(doc)

    if not outline:
        print("PDF doesn't have outline.")
        # tree = build_tree_without_toc(pdf_path)
        # return tree
    else:
        print("PDF has outline.")
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
    # return tree

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract and index PDF structure.")
    # parser.add_argument("pdf_path", help="Path to the PDF file.")
    # args = parser.parse_args()
    pdf_file = "/home/thuyn/pageIndex/PageIndex-Umbrella/tests/pdfs/CDR_Verteidigung_in_der_Tiefe-V1.1_de.pdf"
    # extract_tree(pdf_file)
    doc = fitz.open(pdf_file)
    tree = get_outline(doc)
    with open('treewithinoutline.json', 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2)
    print('finished')

