import os
import json
import fitz  # PyMuPDF

from pageindex import *
from pageindex.page_index_md import md_to_tree


def extract_structure(pdf_path):
    """Trích xuất cấu trúc mục lục (TOC) của PDF bằng pageindex"""
    # Kiểm tra file
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("PDF file phải có phần mở rộng .pdf")
    if not os.path.isfile(pdf_path):
        raise ValueError(f"Không tìm thấy file PDF: {pdf_path}")

    # Cấu hình
    opt = config(
        model="Qwen/Qwen3-8B",
        toc_check_page_num=0,
        max_page_num_each_node=5,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="no",
    )

    # Gọi hàm chính của pageindex
    toc_with_page_number = page_index_main(pdf_path, opt)
    print("✅ Trích xuất cấu trúc PDF hoàn tất.")
    return toc_with_page_number


def add_node_text(pdf_path, toc_data):
    """Thêm nội dung text thực tế vào từng node"""
    print("📄 Đang thêm node_text vào từng mục...")
    doc = fitz.open(pdf_path)
    nodes = toc_data["structure"][0]["nodes"]

    for i, node in enumerate(nodes):
        start_page = node["start_index"] - 1  # fitz index từ 0
        end_page = node["end_index"] - 1

        # Nếu có node tiếp theo → cắt đến trước trang của node sau
        if i < len(nodes) - 1:
            next_start = nodes[i + 1]["start_index"] - 1
            pages_to_extract = range(start_page, next_start)
        else:
            pages_to_extract = range(start_page, end_page + 1)

        text_parts = []
        for p in pages_to_extract:
            if 0 <= p < len(doc):
                text_parts.append(doc[p].get_text("text").strip())

        node_text = "\n".join(text_parts).strip()
        node["node_text"] = node_text

    print("✅ Thêm node_text hoàn tất.")
    return toc_data


def main():
    # 🧠 Đặt đường dẫn PDF ở đây
    pdf_path = "./tests/P4.pdf"

    # Bước 1: Trích xuất cấu trúc mục lục
    toc_data = extract_structure(pdf_path)

    # Bước 2: Thêm node_text
    toc_data = add_node_text(pdf_path, toc_data)

    # Bước 3: Lưu kết quả
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = './tests/results'
    os.makedirs(output_dir, exist_ok=True)

    output_file = f'{output_dir}/{pdf_name}_structure_with_text.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(toc_data, f, ensure_ascii=False, indent=2)

    print(f"🎉 Kết quả đã được lưu tại: {output_file}")


if __name__ == "__main__":
    main()
