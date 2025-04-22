import random
import warnings

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, size, udf, when
from pyspark.sql.types import StringType, ArrayType, LongType

import nltk
from nltk.corpus import words as nltk_words

from transformers import BertTokenizer
from datasets import Dataset, DatasetDict

warnings.filterwarnings("ignore")
nltk.data.clear_cache()
nltk.download('wordnet')  # 下载 wordnet 同义词词库
nltk.download('words')  # 下载英文单词库
eng_words = nltk_words.words()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def deduplicate_data(df):
    """
    Remove duplicate rows from the DataFrame.
    :param df: Input DataFrame.
    :return: DataFrame without duplicates.
    """
    print("Removing duplicates...")
    print("Before deduplication: ", df.count())
    df = df.dropDuplicates()
    print("After deduplication: ", df.count())
    print("=" * 60)
    return df


def filter_data(df):
    """
    Filter the DataFrame based on certain conditions.
    :param df: Input DataFrame.
    :return: Filtered DataFrame.
    """
    print("Filtering data...")
    # --------------------------------------------- #
    # data should have product/price
    # --------------------------------------------- #
    print("Before filtering by price: ", df.count())
    non_numeric_value = df.filter(
        (col("product/price").isNull()) |  # 空值
        (~col("product/price").rlike("^[0-9]*\\.?[0-9]+$"))  # 非数字
    ).select("product/price").distinct()
    non_numeric_value.show()
    df = df.filter(
        (col("product/price").isNotNull()) &  # 空值
        (col("product/price").rlike("^[0-9]*\\.?[0-9]+$"))
    )
    print("After filtering by price: ", df.count())
    # --------------------------------------------- #
    # review text length should between some interval
    # --------------------------------------------- #
    # first we need to count the length of review text
    df = df.filter(col("review/text").isNotNull())  # 空值
    df = df.withColumn("review/text_length", size(split(col("review/text"), " ")))
    # Then we show the statistics of review text length
    length_stats = df.select("review/text_length").describe()
    length_stats.show()
    # Finally we filter the review text length
    print("Before filtering by length: ", df.count())
    df = df.filter(
        col("review/text_length") >= 20
    ).drop("review/text_length")
    print("After filtering by length: ", df.count())
    print("=" * 60)
    return df


def data_augmentation(sentence):
    """
    Easy data augmentation.
    """

    # ramdom deletion
    def random_deletion(words, p):
        if len(words) == 1:
            return words
        if random.uniform(0, 1) < p:
            return words
        return [word for word in words if random.uniform(0, 1) > p]

    # random swap
    def random_swap(words, n):
        length = len(words)
        if length <= 1:
            return words
        for _ in range(n):
            idx1 = random.randint(0, length - 1)
            idx2 = random.randint(0, length - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return words

    # random insertion
    def random_insertion(words, n):
        for _ in range(n):
            new_word = random.choice(eng_words)
            words.insert(random.randint(0, len(words)), new_word)
        return words

    # synonym replacement
    def synonym_replacement(words, n):
        for _ in range(n):
            word = random.choice(words)
            synonyms = nltk.corpus.wordnet.synsets(word)
            if len(synonyms) > 0:
                synonym = random.choice(synonyms).lemmas()[0].name()
                words[words.index(word)] = synonym
        return words

    words = sentence.split(" ")
    change = False
    if random.random() <= 0.2:
        words = random_deletion(words, 0.2)
        change = True
    if random.random() <= 0.2:
        words = random_swap(words, 2)
        change = True
    if random.random() <= 0.1:
        words = random_insertion(words, 2)
        change = True
    if random.random() <= 0.2:
        words = synonym_replacement(words, 2)
        change = True
    if change:
        return " ".join(words)


def augment_data(df):
    print("Before data augmentation: ", df.count())
    augment_udf = udf(data_augmentation, StringType())
    df_with_augmented = df.withColumn("augmented_review", augment_udf("review/text"))
    df_augmented = df_with_augmented.filter(df_with_augmented["augmented_review"].isNotNull())
    df_augmented = df_augmented.drop("review/text").withColumnRenamed("augmented_review", "review/text")
    df_combined = df.union(df_augmented)
    df_combined.show()
    print("After data augmentation: ", df_combined.count())
    return df_combined


def tokenize(sentence):
    tokenized = tokenizer(sentence,
                          truncation=True,
                          add_special_tokens=True,
                          padding="max_length",
                          )
    return tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids']


def tokenize_df(df):
    print("Tokenizing data...")
    tokenize_udf = udf(tokenize, ArrayType(ArrayType(LongType())))
    df_tokenized = df.withColumn("token_ids", tokenize_udf(col("review/text"))) \
        .withColumn("input_ids", col("token_ids")[0]) \
        .withColumn("attention_mask", col("token_ids")[1]) \
        .withColumn("token_type_ids", col("token_ids")[2]) \
        .select("review/score", "input_ids", "attention_mask", "token_type_ids")
    df_tokenized = df_tokenized.withColumn("review/score", when(col("review/score") < 3, 0).otherwise(1))
    df_tokenized.show()
    print("Tokenization complete.")
    print("=" * 60)
    return df_tokenized


def save_df_to_parquet(train_df, test_df):
    # save df to parquet
    train_df.write.parquet(save_path + "train.parquet", mode="overwrite")
    test_df.write.parquet(save_path + "test.parquet", mode="overwrite")


if __name__ == '__main__':
    # spark: （去重） 过滤 数据增强 去噪（太长太短）tokenizer。
    # for unknown reason, import Datasets will cause error in the remote server
    save_path = './'
    amazone_data_path = save_path + "all.csv"
    ss = SparkSession.builder.appName("Preprocessing").master("local[32]"). \
        config("spark.executor.memory", "32g").config("spark.driver.memory", "32g").getOrCreate()
    df = ss.read.csv(amazone_data_path, header=True, sep=',')
    df = deduplicate_data(df)
    df = filter_data(df)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df.show(5, truncate=False)
    # # train_df = augment_data(train_df)
    train_df = tokenize_df(train_df)
    test_df = tokenize_df(test_df)
    dataset = save_df_to_parquet(train_df, test_df)
    # dataset.save_to_disk(save_path + "amazone_dataset")
    ss.stop()

    # convert to hf.dataset in local machine
    train_dataset = Dataset.from_parquet("D:/Data/Amazon Dataset/train.parquet/*.parquet")
    test_dataset = Dataset.from_parquet("D:/Data/Amazon Dataset/test.parquet/*.parquet")
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    dataset_dict.save_to_disk("amazone_dataset")
