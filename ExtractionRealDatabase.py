import json
import gzip


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("parse function is defined")
        
f = open("output.strict", 'w')
print(f)
for l in parse("data\metadata.json.gz"):
    f.write(l + '\n')
print("file opened")
print(l)