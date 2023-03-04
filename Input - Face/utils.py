import json


def write_to_json(json_style_dict, output_path):
    with open(output_path, "w") as fp:
        print(f"Storing data to json at: {output_path}")
        json.dump(json_style_dict, fp)