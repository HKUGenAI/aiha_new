import streamlit as st

import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from google_lens import google_lens

from openai import AzureOpenAI
from dotenv import load_dotenv
import re
import os
import base64


load_dotenv()
data_loader = ImageLoader()

# clip_ef = embedding_functions.OpenCLIPEmbeddingFunction()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3"
)
db_client = chromadb.PersistentClient(os.getenv("CHROMA_DB_PATH"))
md_collection = db_client.get_or_create_collection(name="md_data", embedding_function=sentence_transformer_ef)


llm = AzureOpenAI()
google_lens = google_lens()


def uploadfile_to_base64(img):
    if img is None:
        return None
    binary = base64.b64encode(img.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{binary}"

def get_image_serach_query(image, text_query):
    google_lens_results = google_lens.get_image_results(image_byte = image.getvalue(), num_results=10)
    print(google_lens_results)
    res = llm.chat.completions.create(
        model="trygpt4o",
        messages=[
            {"role": "system", "content": "You are a image search query generator, you will be helping user generate search query based on the provided image, user query and some google lens search results on the image. Include the contex from the google lens search results whe you find this result related to the image, if the image is a person identify the name of the person. **Directly ouput the search query with notion else**."},
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": "Generate a query based on the user query and by describing the image and include text from the image.\n\n User Query: " + text_query + "\n\nGoogle Lens Results: " + "\n".join([result['heading'] for result in google_lens_results])
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": uploadfile_to_base64(image)
                    }
                }
            ] }
        ],
    )

    return res.choices[0].message.content

def get_user_search_query(text_query, history):
    res = llm.chat.completions.create(
        model="trygpt4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a search query generator, you will be helping user generate search query based on the provided user query, and cconversation history. Include the contex from the cconversation history when necessary.\nIf it is a general sentence and a specific query is not needed, output '$none$' exactly."},
            { "role": "user", "content": f"## Conversation History\n{history}\n\n##User query\n{text_query}" }
        ],
    )

    return res.choices[0].message.content


def search(query): 
    search_results = md_collection.query(query_texts=[query], n_results=10)
    md_result = zip(search_results["documents"][0], search_results["metadatas"][0])

    # if image_query:
    #     md_result += zip(search_results["documents"][1], search_results["metadatas"][1])

    text = "<documentList>"
    for doc, meta in md_result:
        def replace_image_links(md_content):
            pattern = r'!\[.*?\]\((.*?)\)'
            return re.sub(pattern, lambda match: f'![image]({match.group(1).replace("chroma","app/static")})', md_content)

        doc = replace_image_links(doc)

        text += f"""<document>
    <title>{meta["resource_name"] + ": " + meta["chunk_title"]}</title>        
    <content>{doc}</content>
    <uri>{meta["resource_name"]}/{meta["original_chunk_id"]}</uri>
</document>"""
    text += "</documentList>"
    return text


def local_image_to_base64(path):
    if path is None:
        return None
    
    with open(path, "rb") as img:
        binary = base64.b64encode(img.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{binary}"


sys_prompt = f"""<systemPrompt>
    <role>You are a AI history research assistant, you will be helping user research on topic on HKU Engineering</role>
    <task>Answer user questions based on the provided document list, provide citations and images whenever you can.</task>
    <documentDescription>Each document contains a title, content and a uri, the content are in markdown format and may include tables and iamges include those in your response when nessacary</documentDescription>

    {{document_list}}

    <answeringStyle>Output in markdown format,use bold emphasis and bullet point when appropriate, and be concise. </answeringStyle>
    <citationStyle>Provide citations for the information you provide in IEEE style, include its index relative to current round of conversation follow by the uri, surround them in []. Never make up your own citations</citationStyle>
</systemPrompt>"""

# Streamed response emulator
def response_generator(sonversation, search_results, image):

    sys_prompt_with_results = sys_prompt.format(document_list=search_results)
    messages = [
            {"role": "system", "content": sys_prompt_with_results},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in sonversation
            ],
        ]
    if image:
        messages[-1] = { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": messages[-1]["content"]
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": uploadfile_to_base64(image)
                    }
                }
            ] } 

    chunks = llm.chat.completions.create(
        model="trygpt4o",
        messages=messages,
        stream=True,
    )

    for chunk in chunks:
        if len(chunk.choices) == 0:
            continue
        chunk_text = chunk.choices[0].delta.content or ""
        yield chunk_text


st.title("AIHA chat")

def search_str_html(search_str):
    return f"<div style='display:inline-block; margin-left: 3.5rem; padding:2px 8px; background-color:#F5F5F5; border-radius: 10px'>🔍 {search_str}</div>"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


    if "image_path" in message:
        col_text, col_image = st.columns(2)
        col_image.image(message["image_path"], caption="Uploaded Image", width=250)
    else:
        col_text = st.container()
    
    if "search_str" in message:
        col_text.html(search_str_html(message["search_str"]))

with st.sidebar:
    # Accept image upload
    uploaded_image = st.file_uploader("Include a photo", type=["jpg", "png", "jpeg"], key=st.session_state["uploader_key"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)


# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if uploaded_image:
        st.session_state.messages[-1]["image_path"] = uploaded_image
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_image:
        search_query = get_image_serach_query(uploaded_image, prompt)
        search_results = search(search_query)
    else:
        search_query = get_user_search_query(prompt, st.session_state.messages)
        if search_query == "$none$":
            search_query = None
            search_results = ""
        else:
            search_results = search(search_query)

    if search_query:
        st.session_state.messages[-1]["search_str"] = search_query

    if uploaded_image:
        col_text, col_image = st.columns(2)
        col_image.image(uploaded_image, caption="Uploaded Image", width=250 )
    else:
        col_text = st.container()
    if search_query:
        col_text.html(search_str_html(search_query))

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(st.session_state.messages, search_results, uploaded_image))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the file uploader
    st.session_state["uploader_key"] += 1
    st.rerun()
