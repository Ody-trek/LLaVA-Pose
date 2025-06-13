import os
import json
import openai
from retrying import retry
from collections import defaultdict

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_complex_reasoning_data():
    try:
        # Replace with COCO original keypoints annotation file path
        with open('annotations.json', 'r') as f:
            coco_data = json.load(f)

        all_generated_data = []

        def save_data():
            if all_generated_data:
                with open('../datasets/generated_data_reasoning.json', 'w') as outfile:
                    json.dump(all_generated_data, outfile)

        @retry(stop_max_attempt_number=3, wait_fixed=2000)
        def request_completion(messages):
            return openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

        grouped_annotations = defaultdict(list)
        for image_id, items in coco_data.items():
            for item in items:
                grouped_annotations[image_id].append(item)

        try:
            for image_id, annotations_group in list(grouped_annotations.items()):
                messages = [{
                    "role": "system", 
                    "content": """
You are an AI visual assistant specializing in analyzing human poses and actions in images. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes and human keypoints, represented as (x1, y1, x2, y2) for bounding boxes and (x, y, visibility) for human keypoints, with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y for bounding boxes, and x, y coordinates along with visibility (0: not labeled, 1: labeled but not visible, 2: labeled and visible) for human keypoints.

The human keypoints represent the following body parts:
1. nose
2. left eye
3. right eye
4. left ear
5. right ear
6. left shoulder
7. right shoulder
8. left elbow
9. right elbow
10. left wrist
11. right wrist
12. left hip
13. right hip
14. left knee
15. right knee
16. left ankle
17. right ankle

The task is to use the provided caption and bounding box/human keypoint information to create a plausible question about the human poses and actions in the image, and provide the answer in detail.

Create complex questions beyond describing the scene. To answer such questions, one should require first understanding the human poses and actions, then based on the background knowledge or reasoning, either explain why the actions are happening that way, or provide guidance and help to the user's request. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first.

**Do not include any coordinates or numerical values in your explanation**. Instead, utilize the data to explain the scene using natural language. Include details like the number of people, their actions, poses, interactions, relative positions, as well as the relationships and interactions between people and objects in the scene. Describe how people are using objects, their proximity to objects, and any activities involving both people and objects.

When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box/human keypoints. Always answer as if you are directly looking at the image.
                    """
                }]

                for item in annotations_group:
                    if 'caption' in item:
                        caption = item.get('caption', '')
                        messages.append({"role": "user", "content": f"caption: {caption}"})

                for item in annotations_group:
                    combined_content = ""
                    for key, value in item.items():
                        if key != 'caption' and key != 'image_id' and key != 'id':
                            combined_content += f"{key}: {value}\n"
                    if combined_content:
                        messages.append({"role": "user", "content": combined_content})

                completion = request_completion(messages)
                
                all_generated_data.append({
                    "image_id": image_id,
                    "completion": completion.choices[0].message.content
                })

        except Exception as e:
            print(f"Error during processing: {e}")
            save_data()

        save_data()

    except Exception as e:
        print(f"Main program error: {e}")

if __name__ == "__main__":
    generate_complex_reasoning_data() 