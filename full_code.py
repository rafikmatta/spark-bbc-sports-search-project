 01 from pyspark.sql.functions import *
 02 from pyspark.conf import SparkConf
 03 from pyspark.context import SparkContext
 04 from pyspark.sql.types import *
 05 from pyspark.sql.functions import *
 06 import re, datetime
 07 
 08 #set the correct number of partitions
 09 sqlContext.setConf("spark.sql.shuffle.partitions", u"8")
 10 
 11 def build_index(doc_folder):
 12     text_files = sc.wholeTextFiles("/user/root/bbcsport/" + doc_folder)
 13     file_count = text_files.count()
 14     # pre-process stage
 15     # 1. remove prefix from file names
 16     hdfs_precursor = "hdfs://sandbox-hdp.hortonworks.com:8020/user/root/bbcsport/"
 17     files = text_files.map(lambda file: (file[0].replace(hdfs_precursor, ""), file[1]))
 18     # 2. clean text, make lower case,
 19     lines = files.map(lambda lines: (lines[0], re.sub('\n+', '\n', lines[1]).replace('\n', ' ')))
 20     lines = lines.map(lambda line: (line[0], re.sub('[^\w\s-]', '', line[1].lower().strip())))
 21     # Map words to docs
 22     words = lines.flatMapValues(lambda word: word.split(" "))
 23     words_df = words.toDF(["file", "word"])
 24     
 25     # calculate TF/IDF
 26     # 1. counts words per doc
 27     file_words_count = words_df.groupBy("file").agg(count("word").alias("word_count"))
 28     # 2. count words overall
 29     words_count = words_df.groupBy("word", "file").count()
 30     # 3.calculate tf
 31     tf = words_count.join(file_words_count, file_words_count.file == words_count.file, 'left').withColumn("tf", col("count") / col("word_count")).select(words_count.word, words_count.file, "tf")
 32     # 4.calculate IDF
 33     doc_freq = words_df.groupBy("word").agg(countDistinct("file").alias("df"))
 34     idf = doc_freq.groupBy("word", "df").agg(log(file_count / column("df")).alias("idf"))
 35     # join data and calculate tf/idf
 36     tfidf = tf.join(idf, tf.word == idf.word, 'left').withColumn("tf_idf", col("tf") * col("idf")).select(tf.word,tf.file,idf.idf,"tf_idf")
 37     # write data to file
 38     tfidf.write.orc(doc_folder + '.orc')
 39     return
 40 
 41 def load_index(index_name):
 42     #read the index file
 43     tfidf = spark.read.orc(index_name + '.orc')
 44     #cache it for use amongst all the queries so it's not reloaded but kept for the whole session
 45     tfidf.persist()
 46     return(tfidf)
 47 
 48 def query_call(query, N,index):
 49     #split the query string into words
 50     query_words = query.lower().split(" ")
 51     #count the total number of words
 52     total_words = len(query_words)
 53     #create the dataframe for the query vec
 54     query_df = spark.createDataFrame(query_words, StringType())
 55     #count the number of times a term occurs in the query string
 56     query_df = query_df.groupBy("value").count().select(col("value").alias("word"), col("count").alias("tf"))
 57     #calculate the IDF of the query and join with the Index to calculate scores
 58     query_idf = query_df.join(broadcast(index), index.word == query_df.word, 'left').select(index.file, query_df.word, query_df.tf,index.idf, index.tf_idf)
 59     #calculate score and order by highest to lowest
 60     results = query_idf.groupBy("file").agg((sum("tf_idf") * (count("word") / total_words)).alias("score")).orderBy(desc("score"))
 61     results.show(N)
 62     return
 63 
 64 #example usage
 65 #build index for rugby
 66 a = datetime.datetime.now()
 67 build_index("rugby")
 68 b = datetime.datetime.now()
 69 print(b-a) #prints time taken
 70 
 71 #load rugby index
 72 a = datetime.datetime.now()
 73 index = load_index("rugby")
 74 b = datetime.datetime.now()
 75 print(b-a)
 76 
 77  ####  This is the code that should be repeated. Once the initial load is done ######
 78 #### The query should run in the same session multiple times ####
 79 a = datetime.datetime.now()
 80 N = 3  # top N results
 81 query = "Yachvili slotted over four penalties"
 82 query_call(query, N,index)
 83 b = datetime.datetime.now()
 84 print(b-a)
