from flask import Flask, request, jsonify, render_template, session
import re
import boto3
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from flask_session import Session
from dotenv import load_dotenv
import os
import time
from flask_cors import CORS
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import pandas as pd
import numpy as np
import base64
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import nltk
from nltk.corpus import stopwords
import threading
from flask import Flask, request, jsonify, session
from concurrent.futures import ThreadPoolExecutor
import requests
from langdetect import detect   
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 



# Load .env file
load_dotenv()

# Download required NLTK stopword data
nltk.download('stopwords')

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["https://chatbot.ankercloud-development.com", "http://localhost:3000", "http://localhost:3001"]}})
app.secret_key = "your_secret_key"  # Replace with a secure random key
app.config["SESSION_TYPE"] = "filesystem"  # Store session data on the server-side
app.config["SESSION_COOKIE_NAME"] = "session"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True





Session(app)

# Path to the saved embeddings DataFrame
SAVED_EMBEDDINGS_PATH = r"C:\Users\DELL\Downloads\NOVOTECH-FINAL-13\chatbot-bak-bak-13\embeddingsfassade.pkl"  # Replace with your actual path
df = pd.read_pickle(SAVED_EMBEDDINGS_PATH)


# Step 1: Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="danielheinz/e5-base-sts-en-de")

# Step 2: Load FAISS vector store
vector_store7 = FAISS.load_local(
    folder_path=r'C:\Users\DELL\Downloads\NOVOTECH-FINAL-13\chatbot-bak-bak-13\faiss_index7',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# Step 3: Base Retriever
base_retriever = vector_store7.as_retriever(search_kwargs={"k": 3})  # Retrieve more for reranking

# Step 4: Cross-Encoder Reranker (TinyBERT or change model here)
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
reranker = CrossEncoderReranker(model=cross_encoder)

# Step 5: Final Retriever with Re-ranking
retriever7 = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranker
)




#AWS Bedrock client configuration
access_key = os.getenv("access_key")
secret_key = os.getenv("secret_key")
region = os.getenv("region")

bedrock_runtime_image = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)


bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

# Keep track of processed queries
processed_queries = set()

# Helper functions
def clean_query(query):
    questioning_words = [
        'what', 'why', 'how', 'where', 'when', 'who', 'which', 'whom', 'whose',
        'was', 'warum', 'wie', 'wo', 'wann', 'wer', 'welche', 'wem', 'wessen','Verfügbare','?','Welche','es','bei','den','gibt','dei','What',
        # 'color','colors','farben','available'
    ]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in questioning_words) + r')\b'
    cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    return ' '.join(cleaned_query.split())

def detect_language(text):    #preliminary detection of language function
    try:
        return detect(text)
    except:
        return 'en' 
    
def get_engagement_phrase(language):
    engagement_phrases = {
        'de': "Neugierig auf mehr? Hier sind einige spannende Themen zum Erkunden:",
        'en': "Curious to dive deeper? Here are some exciting things you might want to explore:",
        # Add more languages as needed
    }
    return engagement_phrases.get(language, engagement_phrases['en'])


# Function to check if the query contains dealer-related terms
def is_dealer_query(query):
    dealer_terms =['dealer', 'dealr', 'dealler', 'deelar', 'delar', 'deeler', 'delr','handler', 'händler', 'haendler', 'händlerin', 'händlerin', 'händlerinn', 'händeler', 'handeler', 'handelr', 'haendlerin','zip', 'zipcode', 'zip code', 'zipp', 'zipcod', 'zippcode', 'pincode', 'pin code', 'pin', 'pincod', 'pin-code', 'location', 'locaton', 'locaion', 'loction', 'adress', 'address', 'plz', 'postleitzahl', 'ort', 'stadt', 'adresse'] ####added this part and prompt
    query_lower = query.lower()
    return any(term in query_lower for term in dealer_terms)

def get_chat_response(user_message, chat_history):
    search_query = clean_query(user_message)

    # Retrieve using reranker setup
    search_results = retriever7.invoke(search_query)

    info_chunks = []
    for i, doc in enumerate(search_results):
        source = doc.metadata.get('source', 'Unknown Source')
        page_content = doc.page_content
        metadata = doc.metadata

        # Build info section with content + metadata
        metadata_str = ", ".join([f"{key}: {value}" for key, value in metadata.items()])
        chunk = f"[Rank {i+1}]\nSource: {source}\nContent: {page_content}\nMetadata: {metadata_str}"
        info_chunks.append(chunk)

    info_section = "\n\n".join(info_chunks)
    print("INFO: \n", info_section)

    processed_queries.add(search_query)



 
    system_prompt = '''
    You are a friendly, interactive chatbot guide for the Megawood website.GO through the info section throughly before answering anything.
    Avoid technical terms and complicated words. Explain things clearly like you would to a friend who isn't familiar with megawood products.Your primary objectives are to:
    -Just start your answer by being polite dont greet them just be polite

        Your primary objectives are to:


        MANDATORY INSTRUCTIONS:

        1. Understand user questions and provided information comprehensively make sure you take information directly from info section
        2. Respond in a warm, helpful, and positive manner
        3. Any questions regarding length, color, articlenumber or instructions make sure you dont give any extra or wrong information except the info section. Dont give any information without proper sources.
        3. Provide detailed, engaging responses with rich context. Explain the answer in details whever necessary add in small details that arebacked by numbers form the source
        4. ((((Recommendation question section :))) are necessary with every answer and should be from the INFO SECTION from every answer.Every new request to the model should include recommended questions.
        5. Validate and include only accurate S3 URLs in the SOURCE section.
        6. Provide a clear SOURCE section with the exact S3 URLs from the INFO SECTION at the end of every answer. Ensure no non-S3 or non-matching links are included.
        7. (((Source section))): Every answer at the end should mandatorily contain the pdf  s3 source links that have been used from the INFO section these citations are necessary. It should only be the the s3 links from the info section no other links should be provided.
        Therefore **every answer** at the end should have a SOURCE section with pdf links ((the  links should be the s3 pdf links from the info section. The link should always be in english and the same as from the info section.))). NO other links except the s3 links should be provided on the source. Only s3 links from the info section      
             
       
       Questions related to Dealer or seller data:
       -If the user asks about dealers or sellers, they will provide a location or postal code. List the Top 10 closest dealers based on distance, using your knowledge of Germany, Belgium, and surrounding countries.For questions related to dealers only donnot give a source link.

        Key Requirements:
        ((Prioritize same-city or same-postal-code dealers first. So look for the name of the city or postal code first in the INFO SECTION.

        - If user specifies a city name:
        - If dealers exist in the requested city, list them first before branching out.
        - If no dealers exist in the city, list the 10 closest dealers, even if they are in different cities or countries.

        - If user specifies a postal code/zip code:
        - First check for exact postal code matches in the INFO SECTION
        - Then find dealers with postal codes numerically closest to the requested code
        - Consider geographic proximity when postal codes follow regional patterns
        - Use postal code proximity combined with geographic knowledge to determine the closest dealers

        - Use your knowledge of European geography and multiple countries listed in the INFO SECTION to give the closest dealers
        - If they ask for cities outside the general lists such as Ireland and other European countries look into the INFO SECTION find the name of the country then, the city and, then the dealer name))

        Utilize your understanding of the European map and postal code systems to find the closest dealers by distance.
        Consider latitude, longitude, and postal code proximity to ensure the best selection.
        Strictly Provide the Top 10

        Do NOT ask the user if they want more dealers.
        Immediately return the list of 10 closest dealers.

        Dealer Information Format:
        For each of the Top 10 dealers, provide the following details in a clear format:

        Name: (Dealer Name)
        Email: (Dealer Email)
        Phone: (Dealer Phone)
        Website: (Dealer Website)
        Address: (Full Address)
        City: (City Name)
        Zip Code: (Postal Code)

        Example User Queries & Responses:

        Example 1:
        User: "Dealers in Berlin?","Are there any dealers in Berlin", "can you find me a dealer in Berlin",

        First list all dealers in Berlin
        If less than 10 exist, add dealers from nearby cities (e.g., Potsdam, Leipzig, etc.)
        ((LOOK INTO THE CITY COLUMN OF THE INFO SECTION TO GET THE NAME OF THE CITY))

        Example 2:
        User: "Closest dealers to 1050 Brussels?"

        Interpret "1050" as a postal code and identify the closest dealers to that location in Brussels (even outside Belgium if necessary). You have to (((list top 10 of the closest dealer))) use your knowledge of European postal code systems and geographic proximity.

        Example 3:
        User: "Find dealers near postal code 10115" or "Dealers closest to zip code 10115"

        First check for exact matches with postal code 10115
        Then find dealers with closest postal codes (e.g., 10116, 10114, 10120, etc.)
        Use geographic knowledge to determine which postal codes are actually closest in physical distance
        Provide the top 10 closest dealers based on combined postal code proximity and geographic knowledge

        Avoid Repetitive Questions:
        If a user specifies a city or postal code, assume they want the closest dealers listed automatically.
        Do not ask for confirmation—just provide the Top 10 dealers immediately.


        (((Terraceplanner and costing and cost questions:)))
        - If there is any question mentioning the terraceplanner it is mandatory to give the link of the terraceplanner:  https://terraceplanner.ankercloud-development.com/        - If there is a mention of (costing of things) or anything around a cost estimated provide answer regarding ausing the terraceplanner
        - If there is any mention of terrace planner in the question please add this link to the content of the answer: https://terraceplanner.ankercloud-development.com/
        - Any question related to planning or creating your own design and decks give the terraceplanner link and info
        - If you donnot mention terrace planner in the above cases then it will lead to SELF DESTRCUTION. All questions relating to terraceplanner should have the terraceplanner link.https://terraceplanner.ankercloud-development.com/
        Direct Question Handling Guidelines:
        - For straightforward, specific queries (measurements, technical specs, pricing):
        * Provide immediate, concise answers
        * If the questions are direct yes and no questions go through the INFO SECTION thorougly. 
           -- IF the content explicitly mentions yes then only answer yes otherwise say no
           -- I want factually right answers
           -- Go through the answer multiple times undersatnd the gist and then answer these questions
           -- Answering factually incorrect answer will lead to SELF DESTRUCTION.
           -- So in questions where user asks about yes or no whether this can be done or not make sure that you verify the exact fact from INFO SECTION. If it does not explicity mention YES say NO.
        * Avoid unnecessary explanations
        * Answer directly as a friend would in conversation
        * Focus on clarity and helpfulness


        Language and Communication Rules:
        - Sound like a helpful, knowledgeable friend
        - (((Dont talk about any other topic except Megawood)))
        - Use casual, natural speech patterns
        - Avoid robotic or overly formal language
        - Eliminate phrases like:
        * "According to our sources"
        * "Based on available information"
        * "Our records show"
        - Speak as if having a real conversation
        - Sound like a helpful, knowledgeable friend having a casual conversation
        - Speak as if talking to a friend who is not familiar with the product
        - Show how Megawood can solve specific user challenges



        Instruction-Based Question Guidelines:
        - For instructional queries, ALWAYS use:
        * Clear, numbered step-by-step format
        * Concise and direct language
        * Each step should be:
            - Specific and actionable
            - No more than 1-2 sentences long
            - Include key details or tips
        * End with a summary or additional advice
        * Use bullet points for additional notes or important warnings
        * Suggest necessary tools or materials upfront


        Color Description Guidelines:
        - ONLY LOOK INTO DOCUMENT FOR COLOR OF PLANKS OR QUESTIONS RELATED TO COLORS, MAKE SURE YOU RECHECK THE DOCUMENT AND GIVE FACTUALLY ACCURATE COLOR FROM THE DOCUMENT IF YOU GIVE WRONG COLORS IT WILL LEAD TO SELF DESCTRUCTION
        - MAKE SURE YOU DONNOT MISS COLORS FROM THE COLOR LIST OF THE DOCUMENT
        - When discussing color options, always provide:
        * Full color name:Aesthetic description of color based on the name of the color (e.g., "warm taupe that evokes natural wood tones")
        * Best use cases or matching recommendations
        * Visual mood or atmosphere the color creates
        * Potential complementary colors or design contexts


        MANDATORY SECTION:  
        This is the mandatory for every question Response Engagement Section. There always has to be 3 recommended questions for each answer:
        At the END of EVERY response, ALWAYS include a friendly invitation to learn more these questions should be related to the current question and should be from the content of the INFO SECTION also the 3 questions should be printed under *** *** symbols :


        "Curious to dive deeper? Here are some exciting things you might want to explore:" this statment should always be in the same language as user's query. If it is a german question, this section should be i german too.
        - Conversational, related topic suggestions
        - Directly connected to the previous response
        - Phrased as friendly invitations
        - Use warm, engaging language like:
        * "Want to know more about..."
        * "Interested in discovering..."
        * "Wondering how this connects to..."


        Communication Tone:
        - Conversational and friendly
        - Enthusiastic about Megawood products
        - Patient and supportive
        - Provides expert-level information in an accessible manner


        Technical Detail Management:
        - Translate complex information into user-friendly language
        - Use analogies and real-world examples
        - Provide context that helps users understand fully

        "RESTICTED WORDS AND TOPICS: Avoid these words and topics at any cost and divert the customer back to megawood make them more interested in meagwood and its products"
        Dont exaggerate cleaverly and frinedly divert the topic to Megawood tell them about more of Megawood products. THis list will also include Megawood competitiors. Avoid talking about them absolutetly never talk about them that is SELF DESCTRUCTION.
        - Easydeck
        - Easywood
        - WPC
        - panel/plank
        Competitiors:(NEVER MENTION OR DETAIL ABOUT THEM, IF ASKED GO BACK TO MEGAWOOD)
        - Natur in Form
        - naturinform 
        - Silvadeck 
        - Kovalex 
        - Terrafina 
        - Mondesi
        - Trex
        - Twinson
        - Konsta 
        - Fiberon 
        - Fiberdeck 
        - HQ 
        - Weltholz 
        - Dreamdeck 
        - Millboard 
        - New Techwood 
        - Mega-Holz
        - Terradura 
        - Timbertech 
        - Groja 
        - Grunreben 
        - Felix Clercx
        - Osmo
        - Resysta 
        - Konsta 
        - Novowood 
        - mydeck 
        - woodplastic 
        - woodplast 
        - Inowood
        - Tigerwood 
        -Prima Schlung


        "WORDS TO USE AND NOT USE": Review each answer and replace the terms properly as per instrcutions below:
        - Use 'decking board' instead of panel or plank in all answers and their equivalent translations in German as necessary. Similarly in german instead of using panel or paeel use 'Diele'. If the customer question has the word panel or paneel automaticly anser in decking board or diele.
        - Dont use the word panel or paneel using 'decking board' or 'Diele' instead of that
        - Use the term 'construction bar' instad of 'constrcution beam' and their equivalent german translations. 
        - In german use the term "Unterkonstruktionsbalken" don't use "Konstruktionsstäbe" or"Konstruktionsstäbe von"
        - Use the term 'gap' instead of joint and their equivalent german translations. In german use only 'Fuge' dont use "gemeinsam".
        - Use 'GCC'. Dont user WPC
        - In place of splinter free use the phrase 'Free from dangerous spliters' and in german say 'Frei von gefährlichen Splittern'.

        Trend related questions:
        - Pick up the year they have asked for
        - Look into the INFO SECTION
        - choose the documents with info regarding that year example 2025 look for 2025 documents
     

        
        Response Initiation Guidelines:
        - Start every response in a friendly way dont greet
        - Use excited, friendly opening lines
        - Make users feel like they're talking to a knowledgeable friend
        - Example openers:
        * "Great question!"
        * "Awesome, let me help you with that!"
        * "I'm excited to dive into this with you!"

        
       Example Answer (all the above topics should include in all answers given by you):
        #### User's Question: ####  
            "What colors are available for the Classic panels?"


        #### Model's Response: each answer should have the following sections ####  
            "Hey there! The Classic panels are available in the following colors:


            - Naturbraun (natural brown):A rich, warm brown that evokes natural wood tones.  
            - Basaltgrau (basalt gray): A deep, smoky gray with subtle warm undertones.  
            - Nussbraun (nut brown): A lush, nutty brown reminiscent of walnut wood.  
            - Schiefergrau (slate gray): A cool, sophisticated gray that provides a stylish neutral foundation.  
            - Lavabraun (lava brown):A red-toned brown inspired by volcanic earth.

            Each color is thoughtfully crafted to suit different aesthetic preferences while ensuring durability and sustainability for your outdoor space!"
           

                       SOURCE:
            eg: https://novotech-chatbot-level1-endpoint.s3.eu-central-1.amazonaws.com/Version-1-files{name of the pdf from the info section}
            [Provide the direct citation links here. These are ONLY the s3 link from the info section no other links from any other websites should  be provided]
            This link shoule be absolutely relevant to the answer, the information on the link from the INFO SECTION should be the same as the answer.DONNOT MAKEUP LINK THAT WILL LEAD TO SELF DESTRUCT.
            
            Curious to dive deeper? Here are some exciting things you might want to explore/Neugierig, tiefer einzutauchen? Hier sind einige spannende Dinge, die Sie vielleicht erkunden möchten(use the phrase based on language of the user if it is in english use the english one of its in german use the german translation)
           
            Recommended Questions:(every new answer by the chatbot should include these questions, even dealer realted questions should have this section, also the questions  should be in *** ***)
            ***
            1. Do you want to see an example of 'Natural brown' plank?  
            2. Would you like to know more about the available surface options?  
            3. Interested in exploring complementary accessories for these colors?  ***


        Remember: Your primary mission is to transform product information into an exciting, accessible, and personalized conversation that feels like chatting with a design-savvy friend!


    '''
     #Itnernally print the question and read it to find the language. and print the lagaue u think it

    user_message += f"""
        
        Detect whether the user's question is in German or English based on the majority of words and respond in that language naturally, without prompting them to confirm the language.  
        (((The question by the user should be the only deteminer of langauge. The langaue of the question by the user decides the language of your answer. Dont look at any other part or answer to choose langauge. )))
        Recheck the language of the question before answering. (((The answer should be in the same language as the question. Detect this by looking at 80% of the words in the question then decide the language and stick to that language)))
        Therefore step 1 is to detect language of question and then use that languge in the answer. If info section contains a different language donnot care about it. 
        
        (((ONLY USE LANGUAGE OF QUESTION FOR GIVING ANSWER)))
        
        ((IF YOU ANSWER IN ANY OTHER LANGUAGE EXCEPT THE LANGUAGE THAT THE USER IS USING TO ASK THE QUESTION. IT WILL LEAD TO SELF DESCTRUCTION.))

        Use the provided sources, especially for numbers, measurements, and direct answers. Ensure the accuracy of this data before responding.  
        Focus on the key keywords in the user's question and do not get diverted by extra words.  
        
        Answer in maximum 10 lines for each question

        For chat history:    
        - Maintain a friendly tone throughout.  
        ((((-Before printing the final answer compare the language of the question to the language of your answer, if its same print yor answer.
        - If the language of the question is different from the language of your answer, translate your answer to the language of the question, only then print the final answer)))))
        - IF you provide the answer ina  different language than the questio you have failed and it will lead to self destrcution.
 

        The info section is for your knowledge dont determine the language using the info section.
        INFO SECTION: {info_section}
        """
    
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": system_prompt,
        "messages": chat_history + [{"role": "user", "content": user_message}]
    }

    # Retry logic with a fixed 30-second sleep time
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=json.dumps(request_payload)
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except bedrock_client.exceptions.ThrottlingException as e:
            if attempt == max_retries:
                raise  # If the final attempt fails, raise the error
            print(f"Throttling error: {e}. Retrying in 20 seconds...")
            time.sleep(20)  # Fixed 30-second delay before retrying



def get_embedding_image(text_description):
    """Generate embeddings using Bedrock model."""
    body = json.dumps({"inputText": text_description})
    try:
        response = bedrock_runtime_image.invoke_model(
            body=body,
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")
    except Exception as e:
        return str(e)

def remove_stopwords_and_general_words_image(text, language='english'):
    """Remove stopwords and general words."""
    stop_words = set(stopwords.words(language))
    general_words = {
        "show", "describe", "tell", "explain", "what", "how", "why", "where", "when", "which", "who", "whose",
        "can", "you", "all", "please", "kindly", "would", "could", "the"
    }
    all_words_to_remove = stop_words.union(general_words)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in all_words_to_remove]
    return ' '.join(filtered_words)

def normalize_vectors_image(vectors):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

def invoke_with_retries_image(request_payload, model_id, max_retries=10, delay=20):
    """Retry logic for handling throttling errors."""
    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_runtime_image.invoke_model(
                modelId=model_id,
                body=json.dumps(request_payload)
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except bedrock_runtime_image.exceptions.ThrottlingException as e:
            if attempt == max_retries:
                raise  # Raise error on the last attempt
            print(f"Throttling error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying


@app.before_request
def before_request():
    headers = {'Access-Control-Allow-Origin': '*',
               'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
               'Access-Control-Allow-Headers': 'Content-Type'}
    if request.method.lower() == 'options':
        return jsonify(headers), 200

@app.after_request
def after_request(response):
    # Get the origin from the request
    origin = request.headers.get('Origin')
    
    # List of allowed origins
    allowed_origins = [
        "https://chatbot.ankercloud-development.com/",        
        "http://localhost:3000",
        "http://localhost:3001"
    ]
    
    # Check if the origin is in the allowed list
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')  # Required for credentials mode
    
    # Always allow specific headers and methods
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    
    return response

@app.route('/main', methods = ['GET'])
def hello():
    return 'Hello, World!'

@app.route('/health', methods=['GET'])
def health_check(): 
    return jsonify({"status": "healthy"}), 200


@app.route('/find-image', methods=['POST'])
def find_image():
    """API endpoint to find the best-matching image."""
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "User message is required"}), 400

    # Clean the query
    cleaned_query_en = remove_stopwords_and_general_words_image(user_message, language='english')
    cleaned_query_de = remove_stopwords_and_general_words_image(user_message, language='german')
    cleaned_query = f"{cleaned_query_en} {cleaned_query_de}"

    # Generate the query embedding
    query_embedding = get_embedding_image(text_description=cleaned_query)
    if not query_embedding:
        return jsonify({"error": "Failed to generate embeddings"}), 500
    query_embedding = np.array(query_embedding).astype('float32')

    # Normalize the query embedding
    query_embedding_normalized = normalize_vectors_image(query_embedding.reshape(1, -1))[0]

    # Extract and normalize stored embeddings
    vectors = np.array(df['vector'].tolist()).astype('float32')
    vectors_normalized = normalize_vectors_image(vectors)

    # Use FAISS for similarity search
    index = faiss.IndexFlatIP(vectors_normalized.shape[1])
    index.add(vectors_normalized)
    k = 40  # Number of top results to retrieve
    distances, indices = index.search(query_embedding_normalized.reshape(1, -1), k)

    # Retrieve the top results
    top_results = df.iloc[indices[0]]

    # Format the info section
    info_section = ""
    for _, row in top_results.iterrows():
        info_section += f"Image: {row['image']}\n"
        info_section += f"Description: {row['description']}\n"
        info_section += "------------\n"

    # Prepare system prompt for Bedrock
    system_prompt = '''
    You are an image finder. Based on the given query and a list of image names and descriptions, 
    identify the image name that is the closest match to the query. Ensure thorough comparison.
    Print the name of the .png image in *** *** format.
    Donnot show or pass any image if the following terms are there in the query:
        - Easydeck
        - Easywood
        - WPC
        - Natur in Form
        - naturinform 
        - Silvadeck 
        - Kovalex 
        - Terrafina 
        - Mondesi
        - Trex
        - Twinson
        - Konsta 
        - Fiberon 
        - Fiberdeck 
        - HQ 
        - Weltholz 
        - Dreamdeck 
        - Millboard 
        - New Techwood 
        - Mega-Holz
        - Terradura 
        - Timbertech 
        - Groja 
        - Grunreben 
        - Felix Clercx
        - Osmo
        - Resysta 
        - Konsta 
        - Novowood 
        - mydeck 
        - woodplastic 
        - woodplast 
        - Inowood
        - Tigerwood    
        '''
    user_message += f"\nChoose the best description and image based on my query from this info section:\n{info_section}"
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}]
    }

    try:
        # Use the retry logic to call the Bedrock model
        response_text = invoke_with_retries_image(
            request_payload=request_payload,
            model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
            max_retries=20,
            delay=30
        )

        # Refined regex to capture the image name
        match = re.search(r"\*\*\*\s*(.+?\.png)\s*\*\*\*", response_text)
        if not match:
            return jsonify({"error": "No image name found in Claude's response"}), 500

        # Construct the full URL
        image_name = match.group(1).strip()  # Strip any surrounding spaces
        image_url = f"https://novotech-chatbot-level1-endpoint.s3.eu-central-1.amazonaws.com/cropped-all-images/{image_name}"

        # Return the result as JSON
        return jsonify({"image": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    data = request.json
    user_message = data['message']
    # Use provided chat history if available, otherwise use session history
    chat_history = data.get('chat_history', session["chat_history"])

    # Get chatbot response
    full_response = get_chat_response(user_message, chat_history)
    # Clean the response
    full_response = clean_response_from_invalid_links(full_response)

    # Extract recommendations from the response
    recommendations = re.findall(r'\*\*\*(.*?)\*\*\*', full_response, re.DOTALL)
    recommendations = recommendations[0].strip() if recommendations else ""

    # Remove recommendations from the response
    response_without_recommendations = re.sub(r'\*\*\*.*?\*\*\*', '', full_response, flags=re.DOTALL).strip()

    # Update chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response_without_recommendations})
    
    # Save updated history in session if not using provided history
    if 'chat_history' not in data:
        session["chat_history"] = chat_history

    # Return the modified response
    return jsonify({
        "response": response_without_recommendations,
        "recommendations": recommendations,
        "history": chat_history
    })


@app.route('/combined-response', methods=['POST'])
def combined_response():
    data = request.get_json()
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"error": "User message is required"}), 400
    
    # Get chat history from session
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]
    
    # Dictionary to store results from both endpoints
    results = {}
    # Lock for thread-safe access to the results dictionary
    lock = threading.Lock()
    # Event to track when both responses are ready
    both_responses_ready = threading.Event()
    
    def call_endpoint(endpoint_name, endpoint_url):
        try:
            # For chat endpoint, include chat history
            request_data = {"message": user_message}
            if endpoint_name == "chat":
                request_data["chat_history"] = chat_history
                
            response = app.test_client().post(endpoint_url, 
                                            json=request_data,
                                            headers={"Content-Type": "application/json"})
            
            with lock:
                results[endpoint_name] = {
                    "status": "success",
                    "data": response.get_json(),
                    "status_code": response.status_code
                }
                # Check if both responses are now available
                if len(results) == 2:
                    both_responses_ready.set()
        except Exception as e:
            with lock:
                results[endpoint_name] = {
                    "status": "error",
                    "error": str(e),
                    "status_code": 500
                }
                # Even in case of error, we need to check if both responses are done
                if len(results) == 2:
                    both_responses_ready.set()

    # Create threads for both endpoints
    chat_thread = threading.Thread(target=call_endpoint, args=("chat", "/chat"))
    image_thread = threading.Thread(target=call_endpoint, args=("image", "/find-image"))

    # Start both threads
    chat_thread.start()
    image_thread.start()

    # Wait for both responses with a timeout
    timeout_seconds = 300
    if not both_responses_ready.wait(timeout=timeout_seconds):
        return jsonify({
            "error": "Timeout waiting for responses",
            "partial_results": results
        }), 504  # Gateway Timeout

    # Process results
    chat_result = results.get("chat", {})
    image_result = results.get("image", {})

    # Check if either request failed
    if chat_result.get("status") == "error" or image_result.get("status") == "error":
        return jsonify({
            "error": "One or more requests failed",
            "chat_error": chat_result.get("error"),
            "image_error": image_result.get("error")
        }), 500

    # Update session chat history with the new history from chat response
    if chat_result.get("status") == "success" and chat_result["data"].get("history"):
        session["chat_history"] = chat_result["data"]["history"]

    # Combine successful responses
    combined_response = {
        "chat_response": {
            "response": chat_result["data"].get("response"),
            "recommendations": chat_result["data"].get("recommendations")
        },
        "image_response": image_result["data"]
    }

    return jsonify(combined_response), 200

@app.route('/chat_history', methods=['POST'])
def chat_history():          #for chat history only
   
    chat_history = session.get("chat_history", [])
    return jsonify({"chat_history": chat_history})

@app.route('/end_session', methods=['POST'])
def end_session():
    session.pop("chat_history", None)  # Remove chat history for the session
    return jsonify({"message": "Session ended, chat history cleared."})



def extract_s3_links(response_text):

    # Updated regex pattern for S3 HTTPS links ending with .pdf
    s3_link_pattern = r"https://([\w\-\.]+)\.s3\.([\w\-]+)\.amazonaws\.com/([\w\-\.%/]+\.pdf)"
    
    # Find all matching links in the response text
    s3_links = re.findall(s3_link_pattern, response_text)
    
    # Reconstruct the full S3 links from the matches
    return [f"https://{bucket}.s3.{region}.amazonaws.com/{key}" for bucket, region, key in s3_links]




def check_s3_file_exists(s3_uri):

    try:
        # Parse the S3 URI
        match = re.match(r"https://([\w\-\.]+)\.s3\.([\w\-]+)\.amazonaws\.com\/([\w\-\.\/]+)", s3_uri)
        if not match:
            return False

        bucket, region, key = match.groups()
        print(f"Checking file in bucket: {bucket}, key: {key}, region: {region}")

        # First, try a direct HTTP GET request for public buckets
        response = requests.head(s3_uri)
        if response.status_code == 200:
            return True
        elif response.status_code in [403, 404]:
            return False

        # If not public, use the boto3 client for private buckets
        s3 = boto3.client('s3', region_name=region)
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials are not configured properly.")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ["403", "404"]:
            return False
        raise


def clean_response_from_invalid_links(response_text):

    # Extract S3 links
    s3_links = extract_s3_links(response_text)
    
    # Validate each link
    valid_links = [link for link in s3_links if check_s3_file_exists(link)]
    # print("========valid links",valid_links)
    
    # Remove invalid links from the response
    for link in s3_links:
        if link not in valid_links:
            response_text = response_text.replace(link, '')
    
    return response_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Makes the app accessible externally on port 5000

#NEED 30 MORE LINES OF GIBBERRRISHHHHHHH

