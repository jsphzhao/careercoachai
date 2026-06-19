import os
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment.")

client = openai.OpenAI(api_key=api_key)

questions = [
    "My landlord didn't tell me about the code violations before I signed my lease. He only told me afterwards, and I think this might be illegal. Is this allowed under the Chicago RLTO?",
    "Does the RLTO provide Chicago tenants the right to terminate their lease due to persistent heating issues when the landlord only provides temporary fixes?",
    "My Chicago lease doesn't expire for another 8 months. But my landlord keeps coming by to show my apartment to prospective tenants, and he insists that this is his right. Is this true?",
    "My landlord just showed up at my door at 7am to do a surprise inspection of my Chicago apartment. Is this legal?",
    "Is it legal for a landlord to evict a tenant in Chicago for having a service animal if the lease prohibits pets?",
    "What rights do Chicago tenants have regarding habitability when living in a cockroach-infested apartment despite multiple exterminator visits?",
    "If a tenant receives a foreclosure notice in Chicago, are they still responsible for rent until the end of the rental agreement?",
    "I talked to a journalist about a Chicago code violation in my apartment, and now my landlord is trying to raise my rent by 50% next month. Does he have grounds to do this?",
    "I joined a local renter's union, but my landlord said that violates the terms of my Chicago lease. This was never addressed, and I'm wondering if he has any grounds to prohibit me from joining such an organization?",
    "I was six days behind on rent and my landlord is terminating my Chicago lease. He hasn't given me the eviction notice yet, but he won't take my unpaid rent. Can I still be evicted if I'm offering to pay the outstanding balance now?",
    "It's been 2 months since I moved out at the end of my Chicago lease, and I still haven't gotten my security deposit back. Is my landlord required to return a security deposit within 45 days and provide a written itemized list of deductions?",
    "A man showed up on my door and insisted that he was my rental property manager, here to fix a leak I called about. However, the landlord never said anything when I started my tenancy. I called my landlord and he confirmed the man's identity, but shouldn't I have been told this earlier, according to the Chicago RLTA?",
    "My Chicago landlord didn't provide me any bed bug information when I moved in, but now I think I have bed bugs. Was he supposed to?",
    "There's a pool of still water in the corner of my bathroom, leaking from the wall. I've called my landlord about it but he says it's my problem. Do I have legal recourse to sue in Chicago?",
    "There was a fire at my Chicago apartment and the windows broke. My landlord hasn't replaced them, but he refuses to lower my rent for the month. What grounds do I have to appeal?",
    "Can a Chicago landlord under the RLTO not tell their tenant that utilities are being stopped, like water, sewer, or trash?",
    "If a landlord in Illinois contacts a tenant about late rent and the tenant threatens legal action for intimidation, could the landlord face any liability under state law?",
    "My landlord changed my locks, even though my Chicago lease is still valid and my rent is paid. Do I have any rights here?",
    "I received a written notice of disturbance several months ago about a noise complaint, and it happened again last night. However, my landlord is ending my rental agreement and giving me one week to move out. Is this the correct timespan for Chicago evictions?",
    "I broke my lease early, and my landlord did find a prospective renter but at a significantly below-market price that he was unwilling to accept. He says that I'm still liable for my rent, even though he technically has a prospective tenant lined up. Is this allowed in Chicago?",
    "Does the RLTO protect me if I'm a student living in university-owned housing?",
    "If I work for my landlord and live on the property as part of my job, does the RLTO still apply to me?",
    "Can my landlord avoid the RLTO by renting for short periods only?",
    "If the building I live in is sold, is the new owner bound by the same rules as my old landlord?",
    "I rent a basement apartment in a 3-flat where the landlord lives upstairs. Am I covered by the RLTO?",
    "I moved in because I work as the building's maintenance person and get free rent. Does RLTO protect me?",
    "I rent a unit in a cooperative building and have a proprietary lease. Does the RLTO apply?",
    "My landlord never told me where my security deposit is being held. Is that allowed?",
    "I found out the building I'm renting in is in foreclosure, but my landlord didn't say anything when I moved in. Can I break the lease?",
    "I signed a lease with someone who didn't tell me who the actual landlord is. Now I have maintenance issues — who's responsible?"
]

results = []

for question in tqdm(questions, desc="Processing Questions", unit="question"):
    responses = []
    
    for _ in range(3):  # set to 3 if you want multiple completions
        try:
            response = client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": (
                        "You are a housing law expert specializing in the Chicago Residential Landlord Tenant Ordinance (RLTO). "
                        "Provide detailed, structured responses with multiple sections, explaining key concepts and legal provisions. "
                        "Cite specific RLTO sections **inline** where applicable. Format your response like a legal guide."
                    )},
                    {"role": "user", "content": question}
                ],
                # This works in SDK v1.x
                timeout=60
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {e}"
        
        responses.append(answer)

    results.append({
        "Question": question,
        "Answer 1": responses[0],
        "Answer 2": responses[1],
        "Answer 3": responses[2]
    })

df = pd.DataFrame(results)
csv_filename = "SPRING_QS_SYS_PROMPT.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8')
print(f"See {csv_filename}")
