from pyspark.sql.functions import *
import re, string
import sys
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


hdfs_precursor = "hdfs://sandbox-hdp.hortonworks.com:8020/user/root/bbcsport/"

spark = SparkSession.builder.appName("BuildIndex").getOrCreate()
sc = SparkContext.getOrCreate()

#get files from HDFS
text_files = sc.wholeTextFiles("/user/root/bbcsport/rugby")
#used for IDF calculation
file_count = text_files.count()

#clean up file name by removing URL
files = text_files.map(lambda file: (file[0].replace(hdfs_precursor, ""), file[1]))
#split text data by line
lines = files.map(lambda lines: (lines[0], lines[1].split("\n")))
#remove punctuation and lower case all the words
lines = lines.map(lambda line: (line[0],[re.sub('[' + string.punctuation + ']', '', curr_line.lower().strip()).strip('') for curr_line in line[1]]))
#flatten into individual lines (1 row per line)
lines = lines.flatMapValues(lambda line: line)
#flatten lines into words (1 row per word)
words = lines.flatMapValues(lambda word: word.split(" "))

#create DF out of the words RDD
words_df = words.toDF(["file", "word"])
#create inverted index
inverted_index = words_df.groupBy("word").agg(collect_list("file").alias("files"))
#count number of words per file
file_words_count = words_df.groupBy("file").agg(count("word").alias("word_count"))
#count number of words
words_count = words_df.groupBy("word", "file").count()
#calculate TF values of each word by joining two dfs and dividing words_count/file_words_count
tf = words_count.join(file_words_count, file_words_count.file == words_count.file, 'left').withColumn("tf", col("count") / col("word_count")).select(words_count.word, words_count.file, "tf")

#count words across all documents
doc_freq = words_df.groupBy("word").agg(countDistinct("file").alias("df"))
#calculate IDF by dividing total doc count by word count over corpus and taking log
idf = doc_freq.groupBy("word", "df").agg(log(file_count / column("df")).alias("idf"))

#calculat TF/IDF by doing multiplication of TF and IDF values
tfidf = tf.join(idf, tf.word == idf.word, 'left').withColumn("tf_idf", col("tf") * col("idf")).select(tf.word,tf.file,idf.idf,"tf_idf")

# write index file in an efficient file format (ORC)
tfidf.write.orc('rugby.orc')

spark.stop()
