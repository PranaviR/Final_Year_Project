import requests
from io import open
import nltk
from nltk.corpus import wordnet

def augment(input_file, output_file):
	with open(output_file, 'w', encoding="utf-8") as fout:
		with open(input_file, 'r') as fin:
			for line in fin:
				print(line)
				line = line.strip()
				linesplit = line.split(" ")
				for token in linesplit:
					response = requests.get('http://api.conceptnet.io/c/en/'+token)
					print('something happened here')
					object = response.json()
					fout.write(token+' ')
					# for edge in object['edges']:
						# if edge['rel']['label'] == 'Synonym' or edge['rel']['label'] == 'RelatedTo' or edge['rel']['label'] == 'IsA':
							# end_label = edge['end']['label'].lower()
							# start_label = edge['start']['label'].lower()
							# if not end_label in items and wordnet.synsets(end_label):
								# fout.write( end_label + ' ')
								# items.add(end_label)
							# if not start_label in items and wordnet.synsets(start_label):
								# fout.write( start_label + ' ')
								# items.add(start_label)
				# fout.write(u'\n')
					all_edges = list()
					for edge in object['edges']:
						if edge['rel']['label'] == 'Synonym' or edge['rel']['label'] == 'RelatedTo' or edge['rel']['label'] == 'IsA':
							edge_items = list()
							edge_items.append(edge['weight'])
							edge_items.append(edge['end']['label'].lower())
							edge_items.append(edge['start']['label'].lower())
							all_edges.append(edge_items)
							
					all_edges.sort(key=lambda x: x[0],reverse=True)
					
					count = 0
					
					items = set()
					items.add(token)
					for i in range(0,len(all_edges)):
						if count >= 5:
							break
						x = all_edges[i][1]
						y = all_edges[i][2]
						if not x in items and wordnet.synsets(x):
							fout.write( x + ' ')
							items.add(x)
							count+=1
						if not y in items and wordnet.synsets(y):
							fout.write( y + ' ')
							items.add(y)
							count+=1
							
				fout.write(u'\n')
	return output_file
							
if __name__ == '__main__':
	augment('auxiliary2.txt','auxiliary_augment2.txt')
				
				

