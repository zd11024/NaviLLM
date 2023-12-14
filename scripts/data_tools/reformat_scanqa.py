import os
import json
import argparse
from tqdm import tqdm

def get_image_metainfo(scene_id, args):
    path = None
    tmp_path = os.path.join(args.image_dir, scene_id)
    if os.path.exists(tmp_path):
        path = tmp_path
    # assert path is not None, f"{scene_id} cannot be None!"
    if path is None:
        raise ValueError(f"{scene_id} cannot be None!")

    image_info, object_info = [], []

    def load_txt(filename):
        pose = []
        with open(filename) as f:
            for line in f.readlines():
                numbers = [float(s) for s in line.strip('\n').split(' ')]
                pose.append(numbers)
        return pose
        

    for filename in os.listdir(os.path.join(path, "color")):
        pose_file = os.path.join(path, "pose", filename.split('.')[0]+".txt")
        if not os.path.exists(pose_file):
            raise ValueError(f"{pose_file} not exist.")

        image_info.append({
            "image_id": filename.split('.')[0],
            "pose": load_txt(pose_file)
        })


    return image_info
        


def main(args):
    for filename in ["ScanQA_v1.0_train.json", "ScanQA_v1.0_val.json", "ScanQA_v1.0_test_w_obj.json", "ScanQA_v1.0_test_wo_obj.json"]:
        with open(os.path.join(args.json_dir, filename)) as f:
            data = json.load(f)

        total_data = len(data)

        new_data = {}
        not_exist = 0
        not_exist_scene_id = {}
        for item in tqdm(data):
            scene_id = item["scene_id"]
            if scene_id in not_exist_scene_id:
                not_exist += 1
                continue

            try:
                if scene_id not in new_data:
                    image_info = get_image_metainfo(scene_id, args)
                    new_data[scene_id] = {
                        "annotation": [],
                        "image_info": image_info,
                    }
            except Exception as e:
                print(f"{e} | SceneId: {scene_id}")
                not_exist += 1
                not_exist_scene_id[scene_id] = 1
                continue

            new_data[scene_id]["annotation"].append({
                "question_id": item["question_id"],
                "question": item["question"],
                "answers": item.get("answers", []),
                "object_ids": item.get("object_ids", []),
                "object_names": item.get("object_names", []),
            })


        data_list = []
        for scene_id, item in new_data.items():
            item["scene_id"] = scene_id
            item['image_info'] = sorted(item['image_info'], key=lambda x: int(x["image_id"]))
            data_list.append(item)
    
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{filename.replace('.json', '')}_reformat.json"), "w") as fout:
            json.dump(data_list, fout)
        
        print(f"Total data: {total_data}")
        print(f"Not exist: {not_exist}")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data/ScanQA")
    args = parser.parse_args()
    main(args)