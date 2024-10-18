from concurrent.futures import ThreadPoolExecutor
from openai import AzureOpenAI
from typing import List, Dict
from tqdm import tqdm 
import pandas as pd
import random
import os

openai_key = os.getenv("OPENAI_API_KEY")
version = '2023-05-15'
endpoint = 'http://localhost:8001'

def sample_by_conversation_id(df, n):

    df['First Number'] = df['Question Set #'].astype(str).str.split('.').str[0]
    unique_first_numbers = df['First Number'].unique()
    sampled_first_numbers = random.sample(list(unique_first_numbers), n)
    sampled_df = df[df['First Number'].isin(sampled_first_numbers)]
    sampled_df = sampled_df.drop(columns=['First Number'])
    sampled_df['query'] = sampled_df['Prompt']
    sampled_df = sampled_df[['Question Set #','conversation_id', 'query']]

    return sampled_df


def gpt4o_chat_completion(
    prompt: str,
) -> str:
    client = AzureOpenAI(
        api_key=openai_key,
        api_version=version,
        azure_endpoint=endpoint,
    )
    messages = [
        {"role": "system", "content": "You are an synthetic data generator."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
        temperature=0,
    )
    return response.choices[0].message.content

def generate_conversation(conversation_history: List[Dict[str, str]]) -> str:

    client = AzureOpenAI(
        api_key=openai_key,
        api_version=version,
        azure_endpoint=endpoint,
    )

    system_prompt = """
    You will be shown a series of user queries. Your task is to generate a new synthetic query that aligns with the given context.

    Follow these instructions carefully when generating the new query:
    1. The new query must be a single sentence.
    2. Ensure it is relevant to the first three original queries.
    3. You are allowed generate topics based on the first three original queries, but stay within the subject area of McKinsey & Company covered industries and capabilities.
    4. Do not add acnronyms or abbreviations not found in the original first three queries.

    Generate the new query below:

    New Query:
    """
    conversation_history.append({
        "role": "system", 
        "content": system_prompt
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        max_tokens=200,
        temperature=0,
    )
    return response.choices[0].message.content


def simulate_conversation_steps(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    simulated_data = df.copy()  
    for conversation_id in tqdm(df['conversation_id'].unique(), desc="Simulating Conversations"):

        conversation = df[df['conversation_id'] == conversation_id]
        # Extract the highest question set number for this conversation
        question_set_max = conversation['Question Set #'].apply(lambda x: float(x)).max()

        conversation_history = [
            {"role": "system", "content": "You are a synthetic query generator."},
        ]

        for _, row in conversation.iterrows():
            conversation_history.append({"role": "user", "content": row['query']})
        # Simulate the next 'steps' responses, generating and appending new queries
        for step in range(steps):
            next_response = generate_conversation(conversation_history)
            # Increment the question set number
            question_set_max += 0.1
            question_set_number = f"{int(question_set_max)}.{str(round(question_set_max % 1, 1))[2:]}" 

            new_row = pd.DataFrame({
                'conversation_id': [conversation_id],
                'query': [next_response],
                'Question Set #': [question_set_number],
            })

            simulated_data = pd.concat([simulated_data, new_row], ignore_index=True)
            conversation_history.append({"role": "user", "content": next_response})

    simulated_data['Question Set #'] = simulated_data['Question Set #'].astype(str)
    simulated_data = simulated_data.sort_values(by=['Question Set #']).reset_index(drop=True)

    return simulated_data


