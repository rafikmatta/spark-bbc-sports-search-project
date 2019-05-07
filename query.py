import sys
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

def query(query_string,N):
    query_words = query_string.lower().split(" ")
    total_words = len(query_words)
    query_df = spark.createDataFrame(query_words,StringType())
    query_df = query_df.groupBy("value").count().select(col("value").alias("word"),col("count").alias("tf"))
    query_idf = query_df.join(broadcast(tfidf), tfidf.word == query_df.word,'left').select(tfidf.file,query_df.word,query_df.tf,tfidf.idf,tfidf.tf_idf)
    results = query_idf.groupBy("file").agg((sum("tf_idf")*(count("word")/total_words)).alias("score")).orderBy(desc("score"))
    results.show(N)


if __name__ == "__main__":

spark = SparkSession\
    .builder\
    .appName("QueryProgram")\
    .getOrCreate()

tfidf = spark.read.orc('rugby.orc')
tfidf.persist()
results = query("Yachvili slotted over over four penalties",10)
spark.stop()