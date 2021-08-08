import json



f = open('./zerowaste_taco_aug/train/labels.json',)
data = json.load(f)


#print(data['annotations'][0])
key_artists = (1 for k in data['annotations'] if k.get('category_id') == 3)



print(sum(key_artists))

print(data['annotations'][-1]['id'])