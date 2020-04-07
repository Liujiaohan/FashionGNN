import os
import json
import os.path as osp


polyvore_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Polyvore', 'raw')
file_trans_name = ["train_no_dup.json","valid_no_dup.json","test_no_dup.json"]

#1. 从 train_no_dump 中过滤出在category_id的套装
dict_list = []
f1 = open(osp.join(polyvore_path,'filter_category_id.txt'), 'r')
line = f1.readline()
category_set = set()
while line:
    l = line.split(' ', 1)
    dict = {}
    dict['id'] = int(l[0])
    category_set.add(int(l[0]))
    dict['name'] = l[1].rstrip("\n")
    dict['frequency'] = 0
    dict_list.append(dict)
    line = f1.readline()
print (category_set)

with open(osp.join(polyvore_path,file_trans_name[0])) as f1:
  all_outfit = json.load(f1)

filtered_training_outfits = []
size = 0
for outfit in all_outfit:
  set_id = outfit["set_id"]
  items = outfit["items"]
  filtered_items = []
  for item in items:
    categoryid = item["categoryid"]
    print (categoryid)
    if int(categoryid) in category_set:
      filtered_items.append(item)
  if len(filtered_items) >= 2:
    size += 1
    outfit_copied = outfit
    outfit_copied["items"] = filtered_items
    filtered_training_outfits.append(outfit_copied)
    print (outfit_copied)
  
  if size >= 200:
    break

filtered_training_outfits_processed = []
for outfit in filtered_training_outfits:
  process_outfit = {
    "items_category": [],
    "items_index": [],
    "set_id": outfit["set_id"]
    }
  for item in outfit["items"]:
    process_outfit["items_category"].append(item["categoryid"])
    process_outfit["items_index"].append(item["index"])
  filtered_training_outfits_processed.append(process_outfit)

with open(osp.join(polyvore_path,"filtered_training_outfits.json"),"w") as f2:
  json.dump(filtered_training_outfits, f2)

with open(osp.join(polyvore_path,"filtered_training_outfits_processed.json"),"w") as f3:
  json.dump(filtered_training_outfits_processed, f3)
 


