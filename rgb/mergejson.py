import json
import os

merge_json_folder = ["DSSDD","RSDD","SSDD"]
root = r"D:\omq\omqdata\sar\dota"
save_path = r"D:\omq\omqdata\sar\dota"

annotation_file = [os.path.join(root,folder,"test.json") for folder in merge_json_folder]

json_contents = []
for ann in annotation_file:
    print(ann)
    with (open(ann, "r") as f):
        json_content = json.load(f)
        # datasets.update(dataset)
        for _num,images in enumerate(json_content["images"]):
            json_content["images"][_num]["file_name"] = \
                    ann.split('\\')[-2]+"\\test\\images\\"+images["file_name"]

        json_contents.append(json_content)
merged_images = []
merged_annotations = []
merged_info = []
merged_class = []
for _json in json_contents:
    merged_images += _json["images"]
    merged_annotations += _json["annotations"]
    merged_info = _json["info"]
    merged_class = _json["categories"]

datasets = {
    "info": merged_info,
    "categories": merged_class,
    "images": merged_images,
    "annotations": merged_annotations

}
with open(save_path+'/test.json', 'w') as json_file:
    json.dump(datasets, json_file)
