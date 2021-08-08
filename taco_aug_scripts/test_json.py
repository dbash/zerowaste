import json 
     
# Data to be written 
f = open('remember_objs.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)
     
with open("remember_objs.json", "w") as outfile: 
    json.dump(data, outfile, indent = 4)