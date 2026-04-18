import os
import json
from PIL import Image


# with open("/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_scenarios.json", 'r') as f:
#     data = json.load(f)
#
#
# for country, lines in data.items():
#     if country == 'ind':
#         print("\n\n")
#         print(country)
#         # print(lines[-2:])
#
#         for i, imagepath in enumerate(lines[-1]['image_path']):
#             image = Image.open(imagepath)
#             image.save(f"{i}.jpg")

with open("/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_scenarios_state.json", "r") as f:
    filtered_scenarios_state = json.load(f)

with open("/weka/home/ext-yingzima/CulturalDrive/traffic_handbook/filtered_us_scenarios_state.json", "r") as f:
    filtered_us_scenarios_state = json.load(f)

print(filtered_us_scenarios_state.keys())

cnt = 0
for country in filtered_scenarios_state.keys():
    print(country)
    print(len(filtered_scenarios_state[country]))
    cnt += len(filtered_scenarios_state[country])

print(cnt)

filtered_scenarios_state['us'] = filtered_us_scenarios_state['us']
#
# with open("filtered_scenarios_state_V1.json", "w") as f:
#     json.dump(filtered_scenarios_state, f, indent=2, ensure_ascii=False)