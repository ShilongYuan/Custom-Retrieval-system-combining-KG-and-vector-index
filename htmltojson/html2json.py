from bs4 import BeautifulSoup as bs
import json
import re
import argparse
import os
from collections import Counter
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input folder path")
parser.add_argument("--output_dir", help="output folder path")
args = parser.parse_args()
# def find_date(texts):
#     for text in texts:
#         if re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text):
#             return re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text).group(0)
#         elif re.search(r'(\d{4})年(\d{1,2})月', text):
#             return re.search(r'(\d{4})年(\d{1,2})月', text).group(0)
#         elif re.search(r'(\d{4})年', text):
#             return re.search(r'(\d{4})年', text).group(0)
#     return ""
def find_date(list_of_strings):
    date_pattern = r"\d{4}.\d{2}.\d{2}"
    for string in list_of_strings:
        if re.search(date_pattern, string):
            l = re.search(date_pattern, string).group()
            l = [l[:4], l[5:7], l[8:10]]
            date = '-'.join(l)
            return date
    return None
def html2json(input_dir, output_dir):
    all_files_processed = []
    files = os.listdir(input_dir)
    PRINT_ONCE = False
    for file in tqdm(files):
        chapter = file.split('.')[1]
        id = file.split('.')[0]
        # if not PRINT_ONCE:
        #     print('chapter: ', chapter)
        soup = bs(open(os.path.join(input_dir, file),encoding='utf-8'), 'html.parser')
        texts = soup.find_all(text=True)
        
        texts = [((i.parent.name if i.parent.name else "") , i.text.strip()) for i in texts if i.text.strip()]
        # if not PRINT_ONCE:
        #     print('texts: ', texts)
        title = ""
        for i in texts:
            if i[0] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                title = i[1]
                break
        if title == "":
            for i in texts:
                if i[0]  == 'title':
                    title = i[1]
                    break
        texts = [i[1] for i in texts]
        date = find_date(texts)
        if len(texts) > 0:
            all_files_processed.append([title, chapter, texts, date,id])
    
        # print('i',i)
    all_files_processed = [{'title': i[0], 'chapter': i[1], 'date': i[3], 'contents': '\n'.join(i[2]),'id':i[4]} for i in all_files_processed]
    # if not PRINT_ONCE:
    #     print('all_files_processed: ', all_files_processed)
    #     PRINT_ONCE = True
    return all_files_processed

processed_data = html2json(args.input_dir, args.output_dir)
hashes = set()
new_processed = []
for i in processed_data:
    if len(''.join(i['contents'])) < 100000 and hash(i['contents']) not in hashes:
        hashes.add(hash(i['contents']))
        new_processed.append(i)
    else:
        pass
print(len(new_processed))

for d in new_processed:
    file_name = str(d['id']) + '.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False))



