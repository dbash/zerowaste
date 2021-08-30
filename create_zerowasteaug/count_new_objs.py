import json


parts = ['train', 'val']

cats = [1,3]

for part in parts:
    f = open('/research/axns2/mabdelfa/ZeroWasteAug/'+ part + '/labels.json',)
    data = json.load(f)
    for cat in cats:
        key_artists = (1 for k in data['annotations'] if k.get('category_id') == cat)
        if cat == 1:
            category = 'plastic'
        elif cat ==3:
            category = 'metal'
        print('{}, {}: {}'.format(part, category, str(sum(key_artists))))

part = 'test'
f = open('/research/axns2/mabdelfa/zero_test/'+ part + '/labels.json',)
data = json.load(f)

for cat in cats:
    key_artists = (1 for k in data['annotations'] if k.get('category_id') == cat)
    if cat == 1:
        category = 'plastic'
    elif cat ==3:
        category = 'metal'
    print('{}, {}: {}'.format(part, category, str(sum(key_artists))))