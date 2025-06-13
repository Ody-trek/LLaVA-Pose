import os
import json
import openai
from collections import defaultdict

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_conversation_data():
    try:
        # Replace with COCO original keypoints annotation file path
        with open('annotations.json', 'r') as f:
            coco_data = json.load(f)

        grouped_annotations = defaultdict(list)
        for item in coco_data['annotations']:
            grouped_annotations[item['image_id']].append(item)

        all_generated_data = []

        def save_data():
            if all_generated_data:
                with open('../datasets/generated_data_conversation.json', 'w') as outfile:
                    json.dump(all_generated_data, outfile)

        try:
            for image_id, annotations_group in grouped_annotations.items():
                fewshot_samples = []

                for item in annotations_group:
                    fewshot_samples.append({
                        'context': f"What can be described in the image?",
                        'response': item['caption']
                    })

                messages = [{"role": "system", "content": """You are an AI visual assistant, and you are seeing a single image. What you see are provided with five sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.

Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers.

Your primary focus should be on generating questions and answers about the poses, actions, and movements of people in the image. Please generate diverse questions that relate primarily to human poses and actions, and provide detailed answers as if you are seeing the image. Only include questions that have definite answers:
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image. Do not ask any question that cannot be answered confidently."""}]

                for sample in fewshot_samples:
                    messages.append({"role": "user", "content": sample['context']})
                    messages.append({"role": "assistant", "content": sample['response']})

                messages.append({"role": "user", "content": "Can you generate some new questions and answers focusing on the poses and actions of people in the image?"})

                completion = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages
                )

                all_generated_data.append({
                    "image_id": image_id, 
                    "conversation": completion.choices[0].message.content
                })

        except Exception as e:
            print(f"Error during processing: {e}")
            save_data()

        save_data()
        
    except Exception as e:
        print(f"Main program error: {e}")

if __name__ == "__main__":
    generate_conversation_data() 
