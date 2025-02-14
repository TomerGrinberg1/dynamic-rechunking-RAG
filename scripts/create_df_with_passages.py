import polars as pl
from spacy import load
from concurrent.futures import ThreadPoolExecutor

# Load spaCy model once
nlp = load("en_core_web_sm")

def divide_with_spacy(text, target_length=200):
    """
    Divide text into passages using spaCy for sentence segmentation.

    :param text: The input document as a string.
    :param target_length: Approximate number of words per passage.
    :return: A list of passages.
    """
    if not text.strip():
        return []  # Skip empty or whitespace-only documents
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    passages, current_passage = [], []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= target_length:
            current_passage.append(sentence)
            current_length += sentence_length
        else:
            passages.append(" ".join(current_passage))
            current_passage = [sentence]
            current_length = sentence_length

    if current_passage:  # Add the last passage if any
        passages.append(" ".join(current_passage))

    return passages

# Read the dataset
df = pl.read_parquet("/home/tomer/learning/dynamic-rechunking-RAG/data/first_5000_rows.parquet")

# Batch process the rows in parallel
def process_batch(rows):
    return [divide_with_spacy(row) for row in rows]

# Split the column into chunks for parallel processing
chunk_size = 1000  # Adjust based on text size and machine capacity
document_chunks = [df["document"][i:i+chunk_size].to_list() for i in range(0, len(df), chunk_size)]

# Process chunks in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(process_batch, document_chunks)

# Flatten the list of results and create a new column
flattened_results = [result for batch in results for result in batch]
df = df.with_columns(pl.Series("document_passages", flattened_results))
df.write_parquet("/home/tomer/dynamic-rechunking-RAG/data/first_5000_rows_with_passeges.parquet")
# Check the resulting columns
print(df.select("document", "document_passages").head(5))