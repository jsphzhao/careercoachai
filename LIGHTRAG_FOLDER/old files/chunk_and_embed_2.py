import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load RLTO txt file
with open("chicago_rlto_v2.txt", "r", encoding="utf-8") as f:
    ordinance_text = f.read()

# Define regex pattern to match section headers precisely
section_pattern = r"(5-12-\d{3}\s+[^.\n]+.*?)\n"  # Captures headers like "5-12-010 Title, purpose and scope"

# Find all section headers with their positions
matches = list(re.finditer(section_pattern, ordinance_text, re.MULTILINE))

sections = []
for i in range(len(matches)):
    start = matches[i].start()
    end = matches[i+1].start() if i + 1 < len(matches) else len(ordinance_text)
    
    # Capture section header and content
    section_text = ordinance_text[start:end].strip()
    sections.append(section_text)

# Chunk large sections into smaller chunks (if needed)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   
    chunk_overlap=100,  
    separators=["\n\n", ". ", "\n"] 
)

# Process and store chunks, keeping track of section headers
chunked_sections = []
for section in sections:
    header_match = re.match(section_pattern, section)  # Extract header
    if header_match:
        header = header_match.group(1)  # Get the section header
        content = section[len(header):].strip()  # Remove header from content

        # Split into sub-chunks but retain header reference
        sub_chunks = text_splitter.split_text(content)
        chunked_sections.extend([f"{header}\n{sub}" for sub in sub_chunks])  # Attach header to each chunk

# Display chunks
for i, chunk in enumerate(chunked_sections[:]):
    print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")

# Total chunks
print(f"# of chunks: {len(chunked_sections)}")


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import openai
from dotenv import load_dotenv  # Import load_dotenv

load_dotenv()

#*init openai embedding model??
# https://platform.openai.com/docs/guides/embeddings

# this is a test key! not sure if i should wait until we get real key?
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")


embedding_model = OpenAIEmbeddings()

#* convert RLTO chunks into LangChain Document format
documents = [Document(page_content=chunk) for chunk in chunked_sections]

#* make FAISS vector store with embeddings
FAISSvec_store = FAISS.from_documents(documents, embedding_model)

#* save the FAISS index 
FAISSvec_store.save_local("rlto_vectorstore")
print("Embeddings stored")
print("\n\n")


#* Load FAISS vector store
vectorstore = FAISS.load_local("rlto_vectorstore", embedding_model)

#* Convert query into embedding, search for relevant sections

questions = [
    "Is a landlord in Illinois required to return a full deposit if a lease was not signed and he changed his mind?",
    "Do Chicago tenants have legal right to break lease if heat keeps breaking & landlord offers only temporary fixes?",
    "Can my landlord in Illinois still evict me if my 5 day notice was taped to my door with an incorrect amount balance shown past due?",
    "I rented an apartment in Illinois and the property manager signed the lease, but not the landlord. Is the lease valid?",
    "My landlord evicted me for having a dog in our apartment. (But it's a service dog.) Is that illegal in Illinois?",
    "What are my rights as a tenant living in a cockroach infested apartment in Illinois? Exterminator has been here at least 4 times.",
    "We got a 60 day notice and found a place before the end of it. Given that we are in Illinois, Do we have to stay until the end and pay rent?",
    "Is it illegal for a landlord to refuse rent in Illinois? My landlord gave me a 5 day notice he now refuses to take money owed.",
    "My landlord says that he can charge a pet deposit and monthly fees for my emotional support animal. Is this true in Illinois?",
    "I rented an apartment in Illinois and the property manager signed the lease, but not the landlord. Is the lease valid?",
    "My landlord in Illinois has not returned my security deposit for more than 45 days now and did not provide me with any estimates.",
    "My roommate left me the apartment. My name is not on the lease. Do I have to move out? I'm in Illinois.",
    "Q: How do I get help filing a complaint against my landlord for shutting off my water and telling me to get out in Illinois",
    "Q: Is it legal for my landlord to have my car towed from the complex? I am behind on rent and planning to move. CHI, IL",
    "Hello, my landlord has kicked me out without 30 day noticed and even threatened to kill me. Is this legal in Illinois?",
    "Can my landlord in Illinois make tenants pay water, sewer and trash not listed in lease to an independent utility company?",
    "My tenant in Illinois said he was going to have me arrested for intimidation so I won't evict him. I messaged him for late rent. Can I go to jail for this?",
    "Can a renter in Illinois claim property from the land lord after living there for twenty years?",
    "My Landlord in Illinois is requiring my roommate and I to pay the outstanding balance left from a previous tenant. Is this legal?",
    "Can a landlord in Illinois sue you if you are just a cosigner on a lease but do not live there"
]

query = questions[3]
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(retrieved_docs):
    print("")
    print("QUERY:", query)
    print("")
    print(f"🔍 Relevant Section {i+1}:\n{doc.page_content}\n{'='*50}")

