import os
import re
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
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

# Define questions
# questions = [
#     "My landlord didn't tell me about the code violations before I signed my lease. He only told me afterwards, and I think this might be illegal. Is this allowed under the Chicago RLTO?",
#     "Does the RLTO provide Chicago tenants the right to terminate their lease due to persistent heating issues when the landlord only provides temporary fixes?",
#     "My Chicago lease doesn't expire for another 8 months. But my landlord keeps coming by to show my apartment to prospective tenants, and he insists that this is his right. Is this true?",
#     "My landlord just showed up at my door at 7am to do a surprise inspection of my Chicago apartment. Is this legal?",
#     "Is it legal for a landlord to evict a tenant in Chicago for having a service animal if the lease prohibits pets?",
#     "What rights do Chicago tenants have regarding habitability when living in a cockroach-infested apartment despite multiple exterminator visits?",
#     "If a tenant receives a foreclosure notice in Chicago, are they still responsible for rent until the end of the rental agreement?",
#     "I talked to a journalist about a Chicago code violation in my apartment, and now my landlord is trying to raise my rent by 50% next month. Does he have grounds to do this?",
#     "I joined a local renter's union, but my landlord said that violates the terms of my Chicago lease. This was never addressed, and I'm wondering if he has any grounds to prohibit me from joining such an organization?",
#     "I was six days behind on rent and my landlord is terminating my Chicago lease. He hasn't given me the eviction notice yet, but he won't take my unpaid rent. Can I still be evicted if I'm offering to pay the outstanding balance now?",
#     "It's been 2 months since I moved out at the end of my Chicago lease, and I still haven't gotten my security deposit back. Is my landlord required to return a security deposit within 45 days and provide a written itemized list of deductions?",
#     "A man showed up on my door and insisted that he was my rental property manager, here to fix a leak I called about. However, the landlord never said anything when I started my tenancy. I called my landlord and he confirmed the man's identity, but shouldn't I have been told this earlier, according to the Chicago RLTA?",
#     "My Chicago landlord didn't provide me any bed bug information when I moved in, but now I think I have bed bugs. Was he supposed to?",
#     "There's a pool of still water in the corner of my bathroom, leaking from the wall. I've called my landlord about it but he says it's my problem. Do I have legal recourse to sue in Chicago?",
#     "There was a fire at my Chicago apartment and the windows broke. My landlord hasn't replaced them, but he refuses to lower my rent for the month. What grounds do I have to appeal?",
#     "Can a Chicago landlord under the RLTO not tell their tenant that utilities are being stopped, like water, sewer, or trash?",
#     "If a landlord in Illinois contacts a tenant about late rent and the tenant threatens legal action for intimidation, could the landlord face any liability under state law?",
#     "My landlord changed my locks, even though my Chicago lease is still valid and my rent is paid. Do I have any rights here?",
#     "I received a written notice of disturbance several months ago about a noise complaint, and it happened again last night. However, my landlord is ending my rental agreement and giving me one week to move out. Is this the correct timespan for Chicago evictions?",
#     "I broke my lease early, and my landlord did find a prospective renter but at a significantly below-market price that he was unwilling to accept. He says that I'm still liable for my rent, even though he technically has a prospective tenant lined up. Is this allowed in Chicago?"
# ]

questions = [
    "My landlord didn't tell me about the code violations before I signed my lease. He only told me afterwards, and I think this might be illegal. Is this allowed under the Chicago RLTO?",
    "Does the RLTO provide Chicago tenants the right to terminate their lease due to persistent heating issues when the landlord only provides temporary fixes?",
    "My Chicago lease doesn't expire for another 8 months. But my landlord keeps coming by to show my apartment to prospective tenants, and he insists that this is his right. Is this true?",
]

# questions = [
#     "Does the RLTO protect me if I’m a student living in university-owned housing?",
#     "If I work for my landlord and live on the property as part of my job, does the RLTO still apply to me?",
#     "Can my landlord avoid the RLTO by renting for short periods only?",
#     "If the building I live in is sold, is the new owner bound by the same rules as my old landlord?",
#     "I rent a basement apartment in a 3-flat where the landlord lives upstairs. Am I covered by the RLTO?",
#     "I moved in because I work as the building’s maintenance person and get free rent. Does RLTO protect me?",
#     "I rent a unit in a cooperative building and have a proprietary lease. Does the RLTO apply?",
#     "My landlord never told me where my security deposit is being held. Is that allowed?",
#     "I found out the building I'm renting in is in foreclosure, but my landlord didn’t say anything when I moved in. Can I break the lease?",
#     "I signed a lease with someone who didn’t tell me who the actual landlord is. Now I have maintenance issues — who’s responsible?"
# ]

# Store results
results = []

# Process each question 3 times with no conversation history
for question in tqdm(questions, desc="Processing Questions", unit="question"):
    responses = []
    
    for i in range(3):  # Ask each question three times     
        # Retrieve relevant RLTO text for each independent query
        query_result = rag.query(question, param=QueryParam(mode="global", only_need_context=True))
        
        # Extract cited sections using regex
        section_pattern = r"§\d{1,2}-\d{2,3}-\d{2,3}"
        cited_sections = set(re.findall(section_pattern, query_result))

        # Generate the final response
        response = openai_client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": (
                    "You are a housing law expert specializing in the Chicago Residential Landlord Tenant Ordinance (RLTO). "
                    "Provide detailed, structured responses with multiple sections, explaining key concepts and legal provisions. "
                    "Cite specific RLTO sections **inline** where applicable. Format your response like a legal guide."
                )},
                {"role": "user", "content": f"{question}\n\nUse the following reference material:\n{query_result}"}
            ],
            timeout=60
        )
        
        # Extract response text
        answer = response.choices[0].message.content
        responses.append(answer)

    # Store results in a structured format
    results.append({
        "Question": question,
        "Answer 1": responses[0],
        "Answer 2": responses[1],
        "Answer 3": responses[2],
    })

# Convert to DataFrame and save to CSV
df = pd.DataFrame(results)
csv_filename = "landlord_tenant_questions_multiple_answers_3.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"✅ Results saved to {csv_filename}")
