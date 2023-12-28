import os

def rename_txt_to_html(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            new_filename = os.path.splitext(filename)[0] + ".html"
            os.rename(os.path.join(input_folder, filename), os.path.join(input_folder, new_filename))

if __name__ == "__main__":
    input_folder = "/Users/weiyuan/RAGproject/raw_data/庆余年"
    rename_txt_to_html(input_folder)
