import subprocess
import os

def fasttext(input_file):
	
	# input_file = "test.txt"
	output_file = "test-vec.txt"
	input_f = open(input_file)
	output_f = open(output_file, "w")
	pretrained_model = '../cc.en.300.bin'
	output = subprocess.call(["./fasttext", "print-sentence-vectors", pretrained_model], stdin=input_f, stdout=output_f)
	input_f.close()
	output_f.close()
	return output_file
		
	
# new_output_file = "test-vec-final.txt"
# with open(new_output_file, 'w') as fout:
	# with open(output_file, 'r') as fin:
		# for line in fin:
				# re.sub("\s+", ",", line.strip())
				# fout.write(line)
				# fout.write('\n')
	
# fin.close()
# fout.close()


# lines = open( 'test-vec.txt', "r" ).readlines()[1::2]
# os.remove("test-vec.txt")
# output_file = "test-vec.txt"
# output_f = open(output_file, "w")
# for line in lines:
	# output_f.write(line)
# output_f.close()
	