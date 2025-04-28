from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

max_pixels = 1280*28*28

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",  
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=max_pixels)


def frame_retrieve(frames, query_relation):
    
    conversation = []
    conversation_base = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": query_relation},
            ],
        }
    ]
    
    for _ in range(len(frames)):
        conversation.append(conversation_base)

    convert_images = []
    a_token_id = 32
    
    for img in frames:
        convert_images.append(Image.fromarray(img))

    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # text_prompt = [
    #     processor.apply_chat_template(con, add_generation_prompt=True)
    #     for con in conversation
    # ]
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=text_prompt, images=convert_images, padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        # Generate logits for the input
        outputs = model(**inputs)
        # Get the logits for the last token in the sequence
        logits = outputs.logits[:, -1, :]
        
        # Extract probabilities for 'A' from the logits
        a_probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, a_token_id].tolist()


    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(a_probabilities, output_text)
        
    return a_probabilities
        
