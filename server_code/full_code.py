from pyspark.sql.functions import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import re, datetime


# read documents

def build_index(doc_folder):
    text_files = sc.wholeTextFiles("/user/root/bbcsport/" + docs_folder)

    # pre-process stage
    # 1. remove prefix from file names
    hdfs_precursor = "hdfs://sandbox-hdp.hortonworks.com:8020/user/root/bbcsport/"
    files = text_files.map(lambda file: (file[0].replace(hdfs_precursor, ""), file[1]))
    # 2. clean text, make lower case,
    lines = files.map(lambda lines: (lines[0], re.sub('\n+', '\n', lines[1]).replace('\n', ' ')))
    lines = lines.map(lambda line: (line[0], re.sub('[^\w\s-]', '', line[1].lower().strip())))

    # Map words to docs
    words = lines.flatMapValues(lambda word: word.split(" "))
    words_df = words.toDF(["file", "word"])
    inverted_index = words_df.groupBy("word").agg(collect_list("file").alias("files"))

    # calculate TF/IDF
    # 1. counts words per doc
    file_words_count = words_df.groupBy("file").agg(count("word").alias("word_count"))
    # 2. count words overall
    words_count = words_df.groupBy("word", "file").count()
    # 3.calculate tf
    tf = words_count.join(file_words_count, file_words_count.file == words_count.file, 'left').withColumn("tf", col("count") / col("word_count")).select(words_count.word, words_count.file, "tf")
    # 4.calculate IDF
    doc_freq = words_df.groupBy("word").agg(countDistinct("file").alias("df"))
    idf = doc_freq.groupBy("word", "df").agg(log(file_count / column("df")).alias("idf"))

    # join data and calculate tf/idf
    tfidf = tf.join(idf, tf.word == idf.word, 'left').withColumn("tf_idf", col("tf") * col("idf")).select(tf.word,tf.file,idf.idf,"tf_idf")

    # write data to file
    tfidf.write.orc(docs_folder + '.orc')
    return


def load_index(index_name):
    tfidf = spark.read.orc(index_name + '.orc')
    tfidf.persist()



def query_call(query, N):
    query_words = query.lower().split(" ")
    total_words = len(query_words)
    query_df = spark.createDataFrame(query_words, StringType())
    query_df = query_df.groupBy("value").count().select(col("value").alias("word"), col("count").alias("tf"))
    query_idf = query_df.join(tfidf, tfidf.word == query_df.word, 'left').select(tfidf.file, query_df.word, query_df.tf,tfidf.idf, tfidf.tf_idf)
    results = query_idf.groupBy("file").agg((sum("tf_idf") * (count("word") / total_words)).alias("score")).orderBy(desc("score"))
    results.show(N)
    return


a = datetime.datetime.now()
build_index("rugby")
b = datetime.datetime.now()
print(a - b)

a = datetime.datetime.now()
load_index("rugby")
b = datetime.datetime.now()
print(a - b)

a = datetime.datetime.now()
N = 1  # top N results
query = "Yachvili slotted over over four penalties"
query_call(query, N)
b = datetime.datetime.now()
print(a - b)
