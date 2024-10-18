from deepeval.metrics import KnowledgeRetentionMetric, ConversationCompletenessMetric, ConversationRelevancyMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI
from typing import List, Dict
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import asyncio

from Conversation_Metrics import *

openai_key = os.getenv("OPENAI_API_KEY")
version = '2023-05-15'
endpoint = 'http://localhost:8001'

df = pd.read_csv('data.csv')

azure_openai = CustomAzureOpenAI(
    openai_key=openai_key,
    version=version,
    endpoint=endpoint,
    temperature=0,
    max_tokens=500
)

results_df = knowledge_retention(
    df,
    model=azure_openai,
    conversation_id_col='conversation_id',
    input_col='query',
    output_col='response',
    threshold=0.5,
    max_workers=5,
    use_completeness=False,
    use_relevancy=False,
    use_knowledge=True
)