import logging
import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from graphframes import GraphFrame
from pyspark.ml.feature import IDF, HashingTF, Tokenizer
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, monotonically_increasing_id, udf
from pyspark.sql.types import FloatType, StringType

from constants import FILE_PATH

load_dotenv()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SIM_THRESHOLD = 92.5


def normalize_name(text) -> str:
    """
    Normalize text by removing spaces, quotes, and replacing digits with their text equivalent

    :param text: Text to normalize
    :return: Normalized text
    """
    if text is None or text == []:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text)
    for d, w in zip(
        "0123456789",
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ],
    ):
        text = text.replace(d, w)
    return text


def cosine_similarity(v1, v2):
    """
    Calculate the cosine simillarity between two vectors

    :param v1: First vector
    :param v2: Second vector
    :return: Cosine similarity multiplied by 100
    """
    # Convert SparseVector to dense array for cosine similarity
    arr1 = np.array(v1.toArray()) if isinstance(v1, SparseVector) else np.zeros(1000)
    arr2 = np.array(v2.toArray()) if isinstance(v2, SparseVector) else np.zeros(1000)
    if np.linalg.norm(arr1) == 0 or np.linalg.norm(arr2) == 0:
        return 0.0
    return float(
        np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)) * 100
    )


def main():
    start_time = time.time()
    logging.info(f"⏱ Deduplication started at {time.ctime(start_time)}")

    # UDFs from functions
    normalize_udf = udf(normalize_name, StringType())
    cosine_udf = udf(cosine_similarity, FloatType())

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("Deduplication with GraphFrames")
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.4-s_2.12")
        .getOrCreate()
    )

    # Set checkpoint needed for GraphFrame
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    # Load data
    df = spark.read.parquet(FILE_PATH)

    # Normalize and increasing ID
    df = df.withColumn("norm_title", normalize_udf(col("product_title"))).withColumn(
        "row_id", monotonically_increasing_id()
    )

    # Tokenize product titles
    tokenizer_title = Tokenizer(inputCol="norm_title", outputCol="tokens_title")
    df = tokenizer_title.transform(df)

    # Generate term frequency vectors
    hashingTF_title = HashingTF(
        inputCol="tokens_title", outputCol="init_title_features", numFeatures=1000
    )
    df = hashingTF_title.transform(df)
    idf_title = IDF(inputCol="init_title_features", outputCol="title_features")
    idf_model_title = idf_title.fit(df)
    df = idf_model_title.transform(df)

    # Use first 5 characters of normalized title for block key to reduce number of matches computed
    df = df.withColumn("block_key", col("norm_title").substr(1, 5))

    # Self-join on block keys
    joined = df.alias("a").join(
        df.alias("b"),
        (col("a.block_key") == col("b.block_key"))
        & (col("a.row_id") < col("b.row_id")),
        "inner",
    )

    # Compute cosine similarity using tf-idfs
    pairs = joined.withColumn(
        "title_cosine", cosine_udf(col("a.title_features"), col("b.title_features"))
    )

    # Filter out low matches
    matches = pairs.filter(col("title_cosine") >= SIM_THRESHOLD).select(
        col("a.row_id").alias("id_a"),
        col("b.row_id").alias("id_b"),
        col("title_cosine"),
    )

    logging.info(f"Number of edges: {matches.count()}")

    # Compute connected components
    vertices = df.select(col("row_id").alias("id"))
    edges = matches.select(col("id_a").alias("src"), col("id_b").alias("dst"))

    g = GraphFrame(vertices, edges)
    components = g.connectedComponents()

    # Join component ID back to original dataframe
    deduped = df.join(components, df.row_id == components.id, "left").drop("id")

    # Get all columns except the group column. Use max to preserve values where possible e.g. price
    agg_cols = [max(c).alias(c) for c in deduped.columns if c != "component"]

    # Group by the component and aggregate
    deduped = deduped.groupBy("component").agg(*agg_cols)

    # Drop unneeded columns
    deduped = deduped.drop(
        "block_key",
        "title_features",
        "init_title_features",
        "tokens_title",
        "norm_title",
    )

    # Save results
    deduped.coalesce(1).write.parquet(
        os.path.join("data", "products_deduped.snappy.parquet")
    )

    end_time = time.time()

    logging.info(f"✅ Deduplication finished at {time.ctime(end_time)}")
    logging.info(f"⏱ Total time elapsed: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
