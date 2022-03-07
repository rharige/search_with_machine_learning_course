import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import nltk
import string
import re
import pandas as pd

#nltk.download('stopwords')
#nltk.download('punkt')

def transform_name(product_name):
    stopwords = nltk.corpus.stopwords.words('english')
    text_new = "".join([i for i in product_name if i not in string.punctuation])
    words = nltk.tokenize.word_tokenize(text_new)
    words = [re.sub("[^0-9A-Za-z\s]", "" , word) for word in words]
    clean_words = [i.lower() for i in words if i not in stopwords]
    stem_words = [nltk.stem.snowball.SnowballStemmer("english").stem(word) for word in clean_words]
    product_name = ' '.join(stem_words)
    return product_name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

records = []
print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      records.append((cat, transform_name(name)))


def filter_by_min_products(df, minimum=50):
    tmp = pd.DataFrame(df.cat.value_counts())
    min_filter = tmp[tmp['cat'] > minimum]
    return df[df.cat.isin(min_filter.index)]
                             
def write_output(df, output_file = r'/workspace/datasets/fasttext/output.fasttext', minimum=50):
    filtered_df = filter_by_min_products(df, minimum)
    with open(output_file, 'w') as output:
        for _, item in filtered_df.iterrows():
            output.write("__label__%s %s\n" % (item['cat'], item['name']))

df = pd.DataFrame(records, columns=['cat', 'name'])
print("Writing results to %s" % output_file)
write_output(df, output_file, minimum=min_products)

