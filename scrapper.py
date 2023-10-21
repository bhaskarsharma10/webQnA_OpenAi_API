import pandas as pd
import openai
import os

from dotenv import load_dotenv
import tiktoken

import numpy as np
from scipy.spatial import distance


load_dotenv()

# Access the API key
openai.api_key = os.getenv("OPENAI_API_KEY")





df=pd.read_csv('processed/embeddings.csv')





def distances_from_embeddings(
    query_embedding,
    embeddings,
    distance_metric="cosine",
):
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": distance.cosine
    }
    distances = distances = [np.dot(query_embedding, embeddings) / (np.linalg.norm(query_embedding) * np.linalg.norm(embeddings)) for e in embeddings]
    return distances

    
    
    

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)



def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    

answer_question(df,'chatgpt')
'''

















import requests
from bs4 import BeautifulSoup

# Define a function to make an HTTP request and retrieve the HTML content
def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return None

# Define a function to scrape and extract specific data from the HTML
def scrape_data(html_content):
    data = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # Use BeautifulSoup to extract specific data
        # Example: titles = soup.find_all("h2")
        # Process and append data to the 'data' list
    except Exception as e:
        print(f"Error while scraping data: {e}")
    return data

# Define a function to save the scraped data to a file or database
def save_data(data):
    # Implement logic to save the data to a file, database, or other storage

# Main function to orchestrate the scraping process
def main():
    target_url = "https://example.com"  # Replace with the URL of the target website
    html_content = get_html(target_url)
    if html_content:
        scraped_data = scrape_data(html_content)
        if scraped_data:
            save_data(scraped_data)

if __name__ == "__main__":
    main() '''