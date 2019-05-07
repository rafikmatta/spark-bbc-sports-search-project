from pyspark.sql.functions import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import re, datetime

#set the correct number of partitions
sqlContext.setConf("spark.sql.shuffle.partitions", u"8")

def build_index(doc_folder):
    text_files = sc.wholeTextFiles("/user/root/bbcsport/" + doc_folder)
    file_count = text_files.count()
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
    tfidf.write.orc(doc_folder + '.orc')
    return

def load_index(index_name):
    #read the index file
    tfidf = spark.read.orc(index_name + '.orc')
    #cache it for use amongst all the queries so it's not reloaded but kept for the whole session
    tfidf.persist()
    return(tfidf)

def query_call(query, N,index):
    #split the query string into words
    query_words = query.lower().split(" ")
    #count the total number of words
    total_words = len(query_words)
    #create the dataframe for the query vec
    query_df = spark.createDataFrame(query_words, StringType())
    #count the number of times a term occurs in the query string
    query_df = query_df.groupBy("value").count().select(col("value").alias("word"), col("count").alias("tf"))
    #calculate the IDF of the query and join with the Index to calculate scores
    query_idf = query_df.join(broadcast(index), index.word == query_df.word, 'left').select(index.file, query_df.word, query_df.tf,index.idf, index.tf_idf)
    #calculate score and order by highest to lowest
    results = query_idf.groupBy("file").agg((sum("tf_idf") * (count("word") / total_words)).alias("score")).orderBy(desc("score"))
    results.show(N)
    return

#example usage
#build index for rugby
a = datetime.datetime.now()
build_index("rugby")
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

#load rugby index
a = datetime.datetime.now()
index = load_index("rugby")
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

##### Run Initial Dummy Query to Actually Load Index #####
a = datetime.datetime.now()
N = 1  # top N results
query = "test"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

####  This is the code that should be repeated. Once the initial load is done ######
#### The query should run in the same session multiple times ####
a = datetime.datetime.now()
N = 1  # top N results
query = "England claim Dubai Sevens glory"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

a = datetime.datetime.now()
N = 3  # top N results
query = "England claim Dubai Sevens glory"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

a = datetime.datetime.now()
N = 5  # top N results
query = "England claim Dubai Sevens glory"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

a = datetime.datetime.now()
N = 1  # top N results
query = "Yachvili slotted over four penalties"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

a = datetime.datetime.now()
N = 3  # top N results
query = "Yachvili slotted over four penalties"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken

a = datetime.datetime.now()
N = 5  # top N results
query = "Yachvili slotted over four penalties"
query_call(query, N,index)
b = datetime.datetime.now()
print("Time taken: " + str((b-a).total_seconds()) + " seconds") #prints time taken
