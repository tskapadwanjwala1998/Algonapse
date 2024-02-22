import requests
from datasets import load_dataset
import csv
"""from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import CompositeTransformation, WordSwapEmbedding
from textattack.augmentation import Augmenter"""


# Hugging Face API token
api_token = "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"  # Replace with your actual API token

HEADERS = {"Authorization": f"Bearer {api_token}"}

def data(symbol,source):
    # Model calls - work on this
    listofmodels = ['textattack/distilbert-base-cased-CoLA',
    'textattack/distilbert-base-cased-MRPC',
    'textattack/distilbert-base-cased-QQP',
    'textattack/distilbert-base-cased-SST-2',
    'textattack/distilbert-base-cased-STS-B',
    'textattack/distilbert-base-uncased-CoLA',
    'textattack/distilbert-base-uncased-MNLI']

    # Load the dataset
    dataset = load_dataset("ag_news", split="train[:1]")
    
    """#trans and constraints
    transformation = CompositeTransformation([
    WordSwapEmbedding()
    # Add more transformations based on the AI/ML model's recommendation
    ])
    constraints = [RepeatModification(), StopwordModification()]
    
	# Run the attack
    attacker = Augmenter(transformation=transformation, constraints=constraints)
    counter = 0 """
    
    

    for example, _ in dataset:
        augmented_texts = symbol
        for augmented_text in augmented_texts:
            csv_filename = f'output_different_models.csv'
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                if csv_file.tell() == 0:
                    csv_writer.writerow(['Model','text', 'label'])
                for model in listofmodels:
                    API_URL = f"https://api-inference.huggingface.co/models/{model}"
                    payload = {"inputs": augmented_text , "options": {"wait_for_model": True}}
                    response = requests.post(API_URL, headers=HEADERS, json=payload)
                    response = response.json()
                    print("Augmented text:", augmented_text)
                    print("Model response:", response)
                    
                    for data in response:
                         for item in data:
                              if item['label'].lower() == 'label_1':
                                print(item['score'])
                                csv_writer.writerow([model,augmented_text, item['score']])

    
    
                    


        

            

    

    

    
    
    




	
