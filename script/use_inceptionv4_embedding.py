import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import os
import os.path as osp
import json

model_name = 'inceptionv4' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

load_img = utils.LoadImage()
# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

polyvore_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Polyvore', 'raw')

filename = "filtered_training_outfits.json"

with open(osp.join(polyvore_path, filename)) as f1:
    all_outfit = json.load(f1)

# images_path = "./images/"
# outfit_list = os.listdir(file_path)

for idx, outfit in enumerate(all_outfit):
  outfit_id = outfit["set_id"]
  items = outfit["items"]
  file_path_outfit = osp.join(polyvore_path, "images" , outfit_id)
  print (idx, len(items))
  for item in items:
    item_id = item["index"]
    file_path_outfit_item = osp.join(file_path_outfit, '{}.jpg'.format(item_id)) 

    input_img = load_img(file_path_outfit_item)
    input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,requires_grad=False)

    output_logits = model(input) # 1x1000

    # img = Image.open(file_path_outfit_item)
    # img = img.resize((229, 229))
    # mat = image.img_to_array(img)
    # mat = np.expand_dims(mat, axis=0)
    # aa = preprocess(mat)
    # if (aa.shape == (1,229,229,1)):
    #     aa = np.tile(aa, (1,1,1,3))
    #     print(aa.shape)
    # itemvector = myinception.predict(aa)

    vector_name = outfit_id + '_' + str(item_id) + '.json'
    with open(osp.join(polyvore_path, 'polyvore_image_vectors' , vector_name), 'w') as f2:
      f2.write(json.dumps(list(output_logits[0].tolist())))