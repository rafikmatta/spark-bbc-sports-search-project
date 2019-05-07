from pyspark.sql.functions import *
import re,string
import sys
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("BuildIndex")\
        .getOrCreate()
	
    sc = SparkContext.getOrCreate()
    text_files = sc.wholeTextFiles("/user/root/bbcsport/rugby")
    file_count = text_files.count()

    hdfs_precursor = "hdfs://sandbox-hdp.hortonworks.com:8020/user/root/bbcsport/"
    files= text_files.map(lambda file: (file[0].replace(hdfs_precursor,""),file[1]))
    lines = files.map(lambda lines: (lines[0],lines[1].split("\n")))

    lines = lines.map(lambda line: (line[0],[re.sub('['+string.punctuation+']', '', curr_line.lower().strip()).strip('') for curr_line in line[1]]))
    lines = lines.flatMapValues(lambda line: line)
    words = lines.flatMapValues(lambda word: word.split(" "))

    words_df = words.toDF(["file","word"])
    inverted_index = words_df.groupBy("word").agg(collect_list("file").alias("files"))
    file_words_count = words_df.groupBy("file").agg(count("word").alias("word_count"))

    words_count = words_df.groupBy("word","file").count()
    tf = words_count.join(file_words_count, file_words_count.file == words_count.file,'left').withColumn("tf", col("count")/col("word_count")).select(words_count.word,words_count.file,"tf")
    #tf = words_count.groupBy("word","file").agg(sum("count").alias("tf"))

    doc_freq = words_df.groupBy("word").agg(countDistinct("file").alias("df"))
    idf = doc_freq.groupBy("word","df").agg(log(file_count/column("df")).alias("idf"))
    tfidf = tf.join(idf, tf.word == idf.word,'left').withColumn("tf_idf", col("tf") * col("idf")).select(tf.word,tf.file,idf.idf,"tf_idf")
    tfidf.write.orc('rugby.orc') # write file in an efficient file format
    spark.stop()

