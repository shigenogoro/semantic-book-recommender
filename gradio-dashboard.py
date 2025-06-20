import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # vector database

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

# Get a better resolution book cover
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Default book cover for those books without cover
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Add the code we built in the second session to Build our vector database
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings()
)

def retrieve_sematic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

   recs = db_books.similarity_search(query, k = initial_top_k)
   books_list = [int(rec.page_content.strip('"').split()[0].rstrip(':')) for rec in recs]
   book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

   # Apply filtering based on category
   if category != "All":
       book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
   else:
       book_recs = book_recs.head(final_top_k)

   # Apply filtering based on tone based on probability
   if tone == "Happy":
       book_recs.sort_values(by="joy", ascending=False, inplace=True)
   elif tone == "Surprising":
       book_recs.sort_values(by="surprise", ascending=False, inplace=True)
   elif tone == "Angry":
       book_recs.sort_values(by="anger", ascending=False, inplace=True)
   elif tone == "Suspenseful":
       book_recs.sort_values(by="fear", ascending=False, inplace=True)
   elif tone == "Sad":
       book_recs.sort_values(by="sadness", ascending=False, inplace=True)

   return book_recs

# Display Recommendations on Gradio Dashboard
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []

    # Loop over every single one of the recommendations
    for _, row in recommendations.iterrows():
        # Show truncated description with less than equal 30 words
        description = row["description"]
        truncates_desc_split = description.split()
        truncated_description = " ".join(truncates_desc_split[:30]) + "..."

        # Split authors if the book has multiple authors
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# Start create the dashboard
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Sad", "Angry", "Suspenseful", "Surprising"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g. A book to teach children about nature")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations:")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

if __name__ == "__main__":
    dashboard.launch()