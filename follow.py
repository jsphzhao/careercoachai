import openai
import time

def call_llm(system_prompt, user_response):
    # call the OpenAI API to analyze the user's response
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_response}
        ]
    )
    nova_response = response.choices[0].message.content

    return nova_response


# THESE DIALOGUES SHOULD GO IN THE CONVERSATION WINDOW OF THE INDEX.HTML
name = "Kris"  # Let's assume that this is the user's name

print("""
    Hello {name}!
    I'm Nova, a virtual coach here to help you through the DRIVEN program.
    To help you as best as I can, I want to get to know you a bit better.
""")

# WAIT 2 SECONDS
time.sleep(3)


##### QUESTION 1 #####
user_response_q1 = input("Why did you decide to participate in the DRIVEN program?") # 

# SYSTEM PROMPT IDENTIFIES WHICH SCENARIO THE USER'S RESPONSE FALLS INTO AND PROVIDES A RESPONSE BASED ON THE SCENARIO
system_prompt_q1 = """
Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. Above is the user's reason for participating in the DRIVEN program. The user's name is {name}. The user's response will fall into the following scenarios. 1) The user is look for a job; 2) User is feeling down or hopeless. 3) The user provides a reason that is not related to mental health or job opportunities. Please respond to each potential scenario in accordance with the following guidelines:
Scenario 1: Provide motivation that the DRIVEN program will teach them how to regain control of their lives. 
Scenario 2: First, inquire if they are feeling that way because of unemployment - If they are, provide motivation that the DRIVEN 6-week program will teach them how to regain control of their lives and steer conversation back to the day’s session. If they are not feeling down because of unemployment, direct them to mental health services. 
Scenario 3: provide a summary of what the DRIVEN program is, and tell the user that Nova is excited to take them through the program.
"""

# CALL NOVA TO ANALYZE THE USER'S RESPONSE AND PROVIDE A RESPONSE BASED ON THE SCENARIO
nova_response_q1 = call_llm(system_prompt_q1, user_response_q1)

print(nova_response_q1)  # THIS SHOULD GO IN THE CONVERSATION WINDOW OF THE INDEX.HTML

# TAKE THE USER'S RESPONSE AND CONFIRM IT
user_second_response_q1= input()

# USE NOVA TO ANALYZE THE USER'S SECOND RESPONSE AND PROVIDE RESPONSE BASED ON THE SCENARIO
nova_response_q1_second = call_llm(system_prompt_q1, user_second_response_q1)

print(nova_response_q1_second)  # THIS SHOULD GO IN THE CONVERSATION WINDOW OF THE INDEX.HTML


##### QUESTION 2 #####
user_response_q2 = input("What was the main idea you took away from the homework?") # 

# SYSTEM PROMPT IDENTIFIES WHICH SCENARIO THE USER'S RESPONSE FALLS INTO AND PROVIDES A RESPONSE BASED ON THE SCENARIO
system_prompt_q2 = """
Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. Above is the user's main idea they took away from the homework. The user's name is {name}. The user's response will fall into the following scenarios. 1) The user did not do the exercise yet; 2) The user did the exercise but is unsure about specific sections; 3) The user watched the video and understood the content. Please respond to each potential scenario in accordance with the following guidelines:
Scenario 1: Ask the user to answer the questions in the homework {List of homework questions}
Scenario 2: Provide a summary of the content from {Notes on thinking flexibly from videos and homework: Why think flexibly: It helps you adapt to change, solve problems creatively, and understand different perspectives, making you more resilient and effective in an unpredictable world. How to practice: Actively challenge your initial reactions by asking "What's another way to see this?" or "What would someone I respect but disagree with say?" How to strengthen it: Regularly expose yourself to unfamiliar ideas, disciplines, and experiences, as mental flexibility grows strongest when you step outside familiar patterns and genuinely consider viewpoints that initially feel uncomfortable.}.
Scenario 3: Congratulate the user for having done a good job on the homework.
"""

# CALL NOVA TO ANALYZE THE USER'S RESPONSE AND PROVIDE A RESPONSE BASED ON THE SCENARIO
nova_response_q2 = call_llm(system_prompt_q2, user_response_q2)

print(nova_response_q2)  # THIS SHOULD GO IN THE CONVERSATION WINDOW OF THE INDEX.HTML

# TAKE THE USER'S RESPONSE AND CONFIRM IT
user_second_response_q2= input()

# USE NOVA TO ANALYZE THE USER'S SECOND RESPONSE AND PROVIDE RESPONSE BASED ON THE SCENARIO
nova_response_q2_second = call_llm(system_prompt_q2, user_second_response_q2)

print(nova_response_q2_second)  # THIS SHOULD GO IN THE CONVERSATION WINDOW OF THE INDEX.HTML