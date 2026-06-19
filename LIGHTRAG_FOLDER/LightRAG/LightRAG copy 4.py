import os
import re
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed
from lightrag.operate import chunking_by_token_size

# Load environment variables
load_dotenv()

# Set up OpenAI API client
openai_client = openai.OpenAI()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

# Custom chunking function to retain section headers
def custom_chunking(content, split_by_character="\n", split_by_character_only=False, chunk_token_size=1000, chunk_overlap_token_size=100, tiktoken_model_name="o1"):
    """Ensures section headers remain attached to the content when chunking."""
    chunks = chunking_by_token_size(content, split_by_character, split_by_character_only, chunk_token_size, chunk_overlap_token_size, tiktoken_model_name)
    
    formatted_chunks = []
    for chunk in chunks:
        lines = chunk["content"].split("\n")
        header = None
        for i, line in enumerate(lines):
            if "§" in line:  # Identify section header
                header = line.strip()
            else:
                if header:
                    lines[i] = f"{header} {line}"  # Attach header to text
                    header = None  # Reset header tracking
        chunk["content"] = "\n".join(lines)
        formatted_chunks.append(chunk)

    return formatted_chunks

# Initialize LightRAG
rag = LightRAG(
    working_dir="./dickens",
    embedding_func=openai_embed,
    llm_model_func=lambda **kwargs: openai_client.chat.completions.create(model="o1", **kwargs),
    chunking_func=custom_chunking
)

# Insert RLTO text
with open("./chicago_rlto_v2.txt") as f:
    rag.insert(f.read())

# Questions to test
questions = [
    "My landlord didn't tell me about the code violations before I signed my lease. He only told me afterwards, and I think this might be illegal. Is this allowed under the Chicago RLTO?",
    "Does the RLTO provide Chicago tenants the right to terminate their lease due to persistent heating issues when the landlord only provides temporary fixes?",
    "My Chicago lease doesn't expire for another 8 months. But my landlord keeps coming by to show my apartment to prospective tenants, and he insists that this is his right. Is this true?",
    "My landlord just showed up at my door at 7am to do a surprise inspection of my Chicago apartment. Is this legal?",
    "Is it legal for a landlord to evict a tenant in Chicago for having a service animal if the lease prohibits pets?",
]

# Store results
response_records = []
chunk_records = []

# Process each question 3 times
for question in tqdm(questions, desc="Processing Questions", unit="question"):
    answers = []

    for repetition in range(1, 4):  # Repeat 3 times
        # Get retrieved context
        query_result = rag.query(question, param=QueryParam(mode="global", only_need_context=True))

        # Save the chunks used (split by double newline)
        chunk_list = [chunk.strip() for chunk in query_result.split("\n\n") if chunk.strip()]
        chunk_row = {"Question": question, "Repetition": repetition}
        for idx, chunk in enumerate(chunk_list):
            chunk_row[f"Chunk {idx + 1}"] = chunk
        chunk_records.append(chunk_row)

        # Extract cited sections
        section_pattern = r"§\d{1,2}-\d{2,3}-\d{2,3}"
        cited_sections = set(re.findall(section_pattern, query_result))

        # Generate model response
        response = openai_client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": (
                    "You are a housing law expert specializing in the Chicago Residential Landlord Tenant Ordinance (RLTO). "
                    "Provide answer first followed by a comprehensive, detailed, structured response with multiple sections, explaining key concepts and legal provisions. "
                    "Most important instruction: Cite specific RLTO sections word-for-word **inline** based on the RLTO where applicable. "
                    "Very important instruction: Do not paraphrase when citing word for word using quotation marks. "
                    "Format your response like a legal guide."
                )},
                {"role": "user", "content": f"{question}\n\nUse the following reference material:\n{query_result}"}
            ],
            timeout=60
        )

        # Store answer
        answer = response.choices[0].message.content
        answers.append(answer)

    # Record final responses
    response_records.append({
        "Question": question,
        "Answer 1": answers[0],
        "Answer 2": answers[1],
        "Answer 3": answers[2],
    })

# Save final responses to CSV
response_df = pd.DataFrame(response_records)
response_df.to_csv("2_newest_landlord_tenant_questions.csv", index=False, encoding='utf-8')
print("✅ Saved: 2_newest_landlord_tenant_questions.csv")

# Save chunk breakdowns to CSV
chunk_df = pd.DataFrame(chunk_records)
chunk_df.to_csv("1_chunks_used_for_responses.csv", index=False, encoding='utf-8')
print("✅ Saved: 1_chunks_used_for_responses.csv")
