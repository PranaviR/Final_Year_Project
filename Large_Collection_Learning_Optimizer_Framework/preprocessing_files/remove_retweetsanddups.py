lines_seen  = set()
outfile = open('HurricaneMatthew.txt', "w")
for line in open('Hurricane_Matthew.txt', "r"):
	if line and line[:4] !='RT @' and line[:2] != 'rt @':
			if line not in lines_seen:
				outfile.write(line)
				lines_seen.add(line)
outfile.close()