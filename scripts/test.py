from datasets import load_dataset
import polars as pl
# dataset = load_dataset("hugosousa/natural_questions_parsed", split="train", streaming=True)

            
            # Convert the dictionary to a Polars DataFrame
# Retrieve and print the first row
# Load the first 1000 rows
# first_5000_rows = []
# short_answer_count = 0

# for i, row in enumerate(dataset):
#     if short_answer_count >= 5000:
#         break
#     if 'short_answers' in row and any(len(inner_list) > 0 for inner_list in row['short_answers']):
#         short_answer_count += 1
#         first_5000_rows.append(row)

# # Convert the list of dictionaries to a Polars DataFrame
# df = pl.DataFrame(first_5000_rows)

# # Save the DataFrame to a Parquet file
# df.write_parquet("/home/tomer/learning/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")
# print(df)
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
