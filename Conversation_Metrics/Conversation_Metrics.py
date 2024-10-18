from deepeval.metrics import KnowledgeRetentionMetric, ConversationCompletenessMetric, ConversationRelevancyMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI
from typing import List, Dict
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import nest_asyncio
import contextlib
import asyncio
import os
import io
nest_asyncio.apply()

def gpt4o_chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    openai_key: str,
    version: str,
    endpoint: str
) -> str:

    client = AzureOpenAI(
        api_key=openai_key,
        api_version=version,
        azure_endpoint=endpoint,
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

class CustomAzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, openai_key, version, endpoint, temperature=0, max_tokens=500):
        self.openai_key = openai_key
        self.version = version
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model_name(self):
        return "Custom Azure OpenAI GPT-4o Model"

    def load_model(self):
        pass

    async def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: gpt4o_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                openai_key=self.openai_key,
                version=self.version,
                endpoint=self.endpoint
            )
        )

    async def a_generate(self, prompt: str) -> str:
        return await self.generate(prompt)

def knowledge_retention(
    df,
    model,
    conversation_id_col='conversation_id',
    input_col='prompt',
    output_col='response',
    threshold=0.5
):
    grouped_items = list(df.groupby(conversation_id_col))

    async def process_conversation(item):
        conversation_id, group = item
        turns = [
            LLMTestCase(
                input=row[input_col],
                actual_output=row[output_col]
            )
            for _, row in group.iterrows()
        ]

        test_case = ConversationalTestCase(turns=turns)
        metric = KnowledgeRetentionMetric(model=model, threshold=threshold)

        turns_text = '\n'.join([
            f"Input: {turn.input}\nOutput: {turn.actual_output}"
            for turn in turns
        ])

        return {
            'conversation_id': conversation_id,
            'conversation': turns_text,
            'knowledge_retention_score': metric.score,
            'reasoning': metric.reason
        }

    results = []
    with tqdm(total=len(grouped_items), desc="Evaluating Knowledge Retention") as pbar:
        for item in grouped_items:
            result = asyncio.run(process_conversation(item))
            results.append(result)
            pbar.update(1)

    results_df = pd.DataFrame(results)

    return results_df

def conversation_relevancy(
    df,
    model,
    conversation_id_col='conversation_id',
    input_col='prompt',
    output_col='response',
    threshold=0.5
):
    grouped_items = list(df.groupby(conversation_id_col))

    def process_conversation(item):
        conversation_id, group = item
        turns = [
            LLMTestCase(
                input=row[input_col],
                actual_output=row[output_col]
            )
            for _, row in group.iterrows()
        ]

        test_case = ConversationalTestCase(turns=turns)
        metric = ConversationRelevancyMetric(model=model, threshold=threshold)

        turns_text = '\n'.join([
            f"Input: {turn.input}\nOutput: {turn.actual_output}"
            for turn in turns
        ])

        return {
            'conversation_id': conversation_id,
            'conversation': turns_text,
            'conversation_relevancy_score': metric.score,
            'reason': metric.reason
        }

    with tqdm(total=len(grouped_items), desc="Evaluating Conversation Relevancy") as pbar:
        results = []
        for item in grouped_items:
            result = process_conversation(item)
            results.append(result)
            pbar.update(1)

    results_df = pd.DataFrame(results)

    return results_df
