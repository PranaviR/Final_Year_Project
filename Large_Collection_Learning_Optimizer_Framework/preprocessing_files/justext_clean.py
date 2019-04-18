import justext
import ast
import json
import re

from io import open

RAW_TEXT_JSON_FILE = 'Sandy_Hook.json'
OUTPUT_FILE = 'Sandy_Hook.txt' 

def get_text_from_json(json_file, output_file):
	with open(output_file, 'w', encoding='utf-8') as fout:
		with open(json_file, 'r', encoding='utf-8') as f:
			for line in f:
				line = json.loads(line)
				cleaned_text = line['text']
				if len(cleaned_text)>0:
					cleaned_text = cleaned_text.replace('\n','')
					cleaned_text = re.sub(r'https?:\/\/[^\s]*', '', cleaned_text, flags=re.MULTILINE)
					fout.write(cleaned_text+'\n')


if __name__ == '__main__':
	get_text_from_json(RAW_TEXT_JSON_FILE, OUTPUT_FILE)
	
