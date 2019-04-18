import re

def process(input_file):
	output_file = "test-vec-final.txt"
	#input_file = "test-vec.txt"
	with open(output_file, 'w') as fout:
		with open(input_file, 'r') as fin:
			for line in fin:
				line = re.sub("\s+", ",", line.strip())
				fout.write(line)
				fout.write('\n')
	fin.close()
	fout.close()
	return output_file