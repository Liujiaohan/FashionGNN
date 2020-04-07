import json
import os.path as osp
"""
filter the 
"""

polyvore_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Polyvore', 'raw')
f1 = open(osp.join(polyvore_path,'filtered_training_outfits.json'), 'r')
train_list = json.load(f1)

dict_list = []
f2 = open(osp.join(polyvore_path, 'filter_category_id.txt'), 'r')
line = f2.readline()
while line:
    l = line.split(' ', 1)
    dict = {}
    dict['id'] = int(l[0])
    dict['name'] = l[1].rstrip("\n")
    dict['frequency'] = 0
    dict_list.append(dict)
    line = f2.readline()


for i in train_list:
    dict = {}
    item_list = i['items']
    #print (type(item_list))
    for j in item_list:
        category_id = j['categoryid']
        for k in dict_list:
            if k['id'] == category_id:
                k['frequency'] += 1

print ('total category: %d'% len(dict_list))

count_100 = 0
dict_list_100 = []
cate_list_100 = []
for i in dict_list:
    if i['frequency'] >= 1:
        dict_list_100.append(i)
        cate_list_100.append(int(i['id']))
        count_100 += 1

print ('more than 100: %d'% count_100)
cate_list_100 = sorted(cate_list_100)
print (cate_list_100)
cid2rcid = {}
for i in range(len(cate_list_100)):
    cid2rcid[int(cate_list_100[i])] = i

with open(osp.join(polyvore_path,"cid2rcid_100.json"), "w") as f2:
    json.dump(cid2rcid, f2)

for outfit in dict_list_100:
    outfit['items'] = []
    for i in train_list:
        items = i['items']
        for j in items:
            if (j['categoryid'] == outfit['id']):
                outfit['items'].append(i["set_id"]+"_"+str(j["index"]))

with open(osp.join(polyvore_path,"category_summarize_100.json"), "w") as f3:
    json.dump(dict_list_100, f3)