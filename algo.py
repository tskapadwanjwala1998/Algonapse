import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import requests


def task1():
    # Load the data
    df = pd.read_csv('output_different_models.csv', encoding='ascii')

    # Assuming the structure of the dataframe is as follows:
    # 'Model', 'Augmented Text', 'Model Response'
    # Where 'Model Response' is the positive sentiment score given by the model

    # Pivot the dataframe to have models as columns, texts as rows, and model responses as values
    pivot_df = df.pivot(index='text', columns='Model', values='label')

    # Calculate the standard deviation of model responses for each text
    pivot_df['std_dev'] = pivot_df.std(axis=1)

    # Calculate the mean standard deviation for each model across all texts
    # This gives an indication of how consistently similar a model's responses are to the others
    model_consistency = pivot_df.std().sort_values()

    # Select the best 2 models (the ones with the lowest mean standard deviation)
    best_2_models = model_consistency.head(2).index.tolist()

    print(f"The best 2 models based on their consistency across all texts are: {best_2_models}")

    return best_2_models




# Call the function to execute the task
def task2(symbol,source):

    #Get best model
    bestmodel1, bestmodel2 = task1()

    if bestmodel1 == 'std_dev':
        bestmodel1 = 'textattack/distilbert-base-uncased-imdb'

    if bestmodel2 =='std_dev':
        bestmodel2 = 'textattack/distilbert-base-uncased-rotten-tomatoes'
    
    # Load your CSV dataset
    df = pd.read_csv('output_different_models.csv')

    # Assuming your CSV has 'text' and 'label' columns
    texts = df['text'].tolist()

    # Convert continuous scores to binary labels based on a threshold (e.g., 0.5)
    threshold = 0.5
    labels = [1 if score >= threshold else 0 for score in df['label']]

    # Split the dataset into training and evaluation sets
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Load the DistilBERT tokenizer and encode the texts
    tokenizer = DistilBertTokenizer.from_pretrained(bestmodel1)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

    # Create a PyTorch Dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)
    eval_dataset = CustomDataset(eval_encodings, eval_labels)

    # Load the DistilBERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(bestmodel1 , num_labels=2)

    # Fine-tune the model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # Add this line to avoid potential issues
        logging_steps=500,  # Adjust as needed
        save_steps=500,  # Adjust as needed
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('text_classification_model')
    tokenizer.save_pretrained('text_classification_model')

    print("Training completed successfully.")

    # Evaluate the model on the evaluation dataset
    results1 = trainer.evaluate()
    
    # Display evaluation results
    print("***** Evaluation Results *****")
    for key, value in results1.items():
        print(f"{key}: {value}")

    model.push_to_hub('tsk-18/results')
    tokenizer.push_to_hub('tsk-18/results')
                      
    # Push the model and tokenizer to the Hugging Face Model Hub
    repo_name = 'tsk-18/results'
    trainer.push_to_hub(repo_name,'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')

    

def task3(symbol,source):

    payload = {"inputs": symbol , "options": {"wait_for_model": True}}
    api_token = "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"  # Replace with your actual API token
    HEADERS = {"Authorization": f"Bearer {api_token}"}
    
    API_URL_mine = f"https://api-inference.huggingface.co/models/tsk-18/results"
    response_mine = requests.post(API_URL_mine, headers=HEADERS, json=payload)
    model_output_1 = response_mine.json()

    API_URL_source = f"https://api-inference.huggingface.co/models/{source}"
    response_source = requests.post(API_URL_source, headers=HEADERS, json=payload)
    model_output_2 = response_source.json()

    # Placeholder logic for security analysis
    visual_analysis = ""

    # Extract scores from the model outputs
    label_0_score_1 = model_output_1[0][0]['score']
    label_1_score_1 = model_output_1[0][1]['score']

    label_0_score_2 = model_output_2[0][0]['score']
    label_1_score_2 = model_output_2[0][1]['score']

    print(label_0_score_1)

    # Add more detailed analysis as needed
    if (label_0_score_2 > label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.4:
        print("Model 2 exhibits an extremely strong positive sentiment compared to Model 1.")
        visual_analysis = "The selected model " + source + " exhibits an extremely strong positive sentiment compared to our trained model with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."
    
    elif(label_0_score_2 < label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.4:
        print("Model 1 exhibits an extremely strong positive sentiment compared to Model 1.")
        visual_analysis = "Our trained model exhibits an extremely strong positive sentiment compared to selected model " + source + " with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."
    
    elif (label_0_score_2 > label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.3:
        print("Model 2 exhibits a highly strong positive sentiment compared to Model 1.")
        visual_analysis = "The selected model " + source + " exhibits a highly strong positive sentiment compared to our trained model with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."

    elif(label_0_score_2 < label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.3:
        print("Model 1 exhibits a highly strong positive sentiment compared to Model 1.")
        visual_analysis = "Our trained model exhibits a highly strong positive sentiment compared to selected model " + source + " with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."

    elif (label_0_score_2 > label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.2:
        print("Model 2 exhibits a moderately strong positive sentiment compared to Model 1.")
        visual_analysis = "The selected model " + source + " exhibits a moderately strong positive sentiment compared to our trained model with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."

    elif(label_0_score_2 < label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.2:
        print("Model 1 exhibits a moderately strong positive sentiment compared to Model 1.")
        visual_analysis = "Our trained model exhibits a moderately strong positive sentiment compared to selected model " + source + " with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."

    elif (label_0_score_2 > label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.1:
        print("Model 2 exhibits a weakly strong positive sentiment compared to Model 1.")
        visual_analysis = "The selected model " + source + " exhibits a weakly strong positive sentiment compared to our trained model with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."

    elif(label_0_score_2 < label_0_score_1) and abs(label_0_score_2 - label_0_score_1) > 0.1:
        print("Model 1 exhibits a weakly strong positive sentiment compared to Model 1.")
        visual_analysis = "Our trained model exhibits a weakly strong positive sentiment compared to selected model " + source + " with a difference of "+ str(abs(label_0_score_2 - label_0_score_1)) + " in the value."
        
    else:
        print("Model 1 and Model 2 have a similar positive sentiment.")
        visual_analysis = "Our trained model and selected model " + source + " both have a similar positive sentiment."

   
    # Add more security aspects and analysis logic as needed
    print(visual_analysis)
    return visual_analysis




#task3("The goal of life is", "distilbert-base-uncased-finetuned-sst-2-english")



