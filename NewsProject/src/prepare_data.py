import pandas as pd

# File paths
train_path = "../dataset/train.csv"
test_path = "../dataset/test.csv"

# AG News files have no column names
columns = ["label", "title", "description"]

# Read the datasets
train_df = pd.read_csv(train_path, names=columns)
test_df = pd.read_csv(test_path, names=columns)

# Combine train and test
df = pd.concat([train_df, test_df], ignore_index=True)

# Convert numeric labels to category names
category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

df["category"] = df["label"].map(category_names)

# Combine title + description into one text column
df["text"] = df["title"] + " " + df["description"]

# Keep only the columns we need
df = df[["text", "category"]]

# Save the cleaned dataset
df.to_csv("../dataset/news.csv", index=False)

print("Dataset prepared successfully!")
print("Total articles:", len(df))
print("\nCategory distribution:")
print(df["category"].value_counts())