import json, pprint, requests, textwrap,time
from flask import Flask, render_template, request, jsonify
import atexit

host = 'http://localhost:8999'
data = {'kind': 'pyspark'}
headers = {'Content-Type': 'application/json'}
location = None
session_url = None

def init_spark():
    r = requests.post(host+'/sessions', data=json.dumps(data), headers=headers)
    location = r.headers['location']
    session_url = host + location
    return(session_url,location)

def poll_session_status(session_url):
    r = requests.get(session_url, headers=headers)
    return(r.json())

def build_index(folder, location):
    statements_url = host + location + '/statements'
    data = {
      'code': textwrap.dedent("""
    from pyspark.sql.functions import *
    import re,string
    import sys
    from pyspark.sql.types import *
    from pyspark.sql.functions import *
    text_files = sc.wholeTextFiles("/user/root/bbcsport/""" + folder + """)
    file_count = text_files.count()
    hdfs_precursor = "hdfs://sandbox-hdp.hortonworks.com:8020/user/root/bbcsport/"
    files= text_files.map(lambda file: (file[0].replace(hdfs_precursor,""),file[1]))
    lines = files.map(lambda lines: (lines[0],lines[1].split('\\n')))
    lines = lines.map(lambda line: (line[0],[re.sub('['+string.punctuation+']', '', curr_line.lower().strip()).strip('') for curr_line in line[1]]))
    lines = lines.flatMapValues(lambda line: line)
    words = lines.flatMapValues(lambda word: word.split(" "))
    words_df = words.toDF(["file","word"])
    inverted_index = words_df.groupBy("word").agg(collect_list("file").alias("files"))
    file_words_count = words_df.groupBy("file").agg(count("word").alias("word_count"))
    words_count = words_df.groupBy("word","file").count()
    tf = words_count.join(file_words_count, file_words_count.file == words_count.file,'left').withColumn("tf", col("count")/col("word_count")).select(words_count.word,words_count.file,"tf")
    doc_freq = words_df.groupBy("word").agg(countDistinct("file").alias("df"))
    idf = doc_freq.groupBy("word","df").agg(log(file_count/column("df")).alias("idf"))
    tfidf = tf.join(idf, tf.word == idf.word,'left').withColumn("tf_idf", col("tf") * col("idf")).select(tf.word,tf.file,idf.idf,"tf_idf")
    tfidf.write.orc('rugby.orc')
        """)
    }
    r = requests.post(statements_url, data=json.dumps(data), headers=headers)
    return(r.headers['location'])

def query_call(query,N):
    statements_url = host + location + '/statements'
    query = "'" + query + "'"
    data = {
        'code': textwrap.dedent("""
        import sys
        from pyspark.sql.types import *
        from pyspark.sql.functions import *
        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession
        tfidf = spark.read.orc('rugby.orc')
        tfidf.persist()
        query_string = """ + query +
        """ 
        N = """ + str(N) +
        """
        query_words = query_string.lower().split(" ")
        total_words = len(query_words)
        query_df = spark.createDataFrame(query_words,StringType())
        query_df = query_df.groupBy("value").count().select(col("value").alias("word"),col("count").alias("tf"))
        query_idf = query_df.join(broadcast(tfidf), tfidf.word == query_df.word,'left').select(tfidf.file,query_df.word,query_df.tf,tfidf.idf,tfidf.tf_idf)
        results = query_idf.groupBy("file").agg((sum("tf_idf")*(count("word")/total_words)).alias("score")).orderBy(desc("score"))
        results.show(N)
        """)
    }
    r = requests.post(statements_url, data=json.dumps(data), headers=headers)
    return (r)

def check_call(location):
    statement_url = host + location
    r = requests.get(statement_url, headers=headers)
    return(r.json())

app = Flask(__name__)

@app.route("/query",methods=('GET', 'POST'))
def do_query():
    data = {}
    if request.method == 'POST':
        query = request.form['query']
        amount = request.form['amount']
        error = None
        if not query:
            error = 'Query is required.'
        elif not amount:
            error = 'Amount is required.'
        else:
            result = query_call(query,amount)
            location_temp = result.headers['location']
            data_temp = result.json()
            while data_temp['state'] != 'available':
                result = check_call(location_temp)
                data_temp = result
            data = result['output']['data']['text/plain']
    return jsonify(data)

@app.route("/stop",methods=('GET', 'POST'))
def stop_spark():
    requests.delete(session_url, headers=headers)

@app.route("/",methods=('GET', 'POST'))
def main_page():
    return render_template('index.html')

if __name__ == '__main__':
    session_url_local,location_check = init_spark()
    session_url = session_url_local
    waiting = True
    print("Starting server...")
    result = poll_session_status(session_url)
    while waiting:
        time.sleep(15)
        result = poll_session_status(session_url)
        state = result['state']
        if state == 'starting':
            print("Going...")
        else:
            print("Started")
            waiting = False
    location = location_check
    app.run()