import os
import json
import random
import openai
from retrying import retry
from collections import defaultdict

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_detailed_description_data():
    try:
        # Replace with COCO original keypoints annotation file path
        with open('annotations.json', 'r') as f:
            coco_data = json.load(f)

        all_generated_data = []

        def save_data():
            if all_generated_data:
                with open('../datasets/generated_data_detailed.json', 'w') as outfile:
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

        detailed_description_sentences = [
            "Describe the actions and poses of people in the following image in detail.",
            "Provide a detailed description of the poses of people in the given image.",
            "Explain the various details of the actions of people you see in the image.",
            "Share a comprehensive analysis of the behaviors of people presented in the image.",
            "Offer a thorough analysis of the actions of people in the image.",
            "Explain the various poses and actions of people in the displayed image with great detail.",
            "Characterize the poses of people in the image using a well-detailed description.",
            "Break down the actions of people in the image in a detailed manner.",
            "Walk through the important details of the actions of people in the image.",
            "Portray the poses and actions of people in the image with a rich, descriptive narrative.",
            "Narrate the actions and poses of people in the image with precision.",
            "Analyze the poses and actions of people in the image in a comprehensive and detailed manner.",
            "Illustrate the actions and poses of people in the image through a descriptive explanation.",
            "Examine the actions and poses of people in the image closely and share their details.",
            "Write an exhaustive depiction of the actions of people in the given image.",
            "Carefully observe the people in the image and share the details of their poses and actions."
        ]

        try:
            for image_id, annotations_group in list(grouped_annotations.items()):
                messages = [{
                    "role": "system", 
                    "content": """
You are an AI visual assistant specializing in analyzing human poses and actions in images. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes and human keypoints, represented as (x1, y1, x2, y2) for bounding boxes and (x, y, visibility) for keypoints, with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y for bounding boxes, and x, y coordinates along with visibility (0: not labeled, 1: labeled but not visible, 2: labeled and visible) for keypoints.

The keypoints represent the following body parts:
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

Using the provided captions and bounding box/human keypoint information, describe the scene in a detailed manner, focusing on the human poses and actions.

Instead of directly mentioning the bounding box or keypoint coordinates, utilize this data to explain the scene using natural language. Include details like the number of people, their actions, poses, interactions, and relative positions.

When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box/keypoints. Always answer as if you are directly looking at the image.
                    """
                }]

                for item in annotations_group:
                    if 'caption' in item:
                        caption = item.get('caption', '')
                        messages.append({"role": "user", "content": f"caption: {caption}"})

                for item in annotations_group:
                    combined_content = ""
                    if 'bbox' in item:
                        combined_content += f"person: {item['bbox']}\n"
                    if 'keypoints' in item:
                        combined_content += f"Keypoints: {item['keypoints']}\n"
                    
                    if combined_content:
                        messages.append({"role": "user", "content": combined_content})

                user_message = random.choice(detailed_description_sentences)
                messages.append({"role": "user", "content": user_message})

                completion = request_completion(messages)
                
                all_generated_data.append({
                    "image_id": image_id,
                    "user_message": user_message,
                    "completion": completion.choices[0].message.content
                })

        except Exception as e:
            print(f"Error during processing: {e}")
            save_data()

        save_data()

    except Exception as e:
        print(f"Main program error: {e}")

if __name__ == "__main__":
    generate_detailed_description_data() 
