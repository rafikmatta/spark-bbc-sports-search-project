# BBC Sports Search Engine Project using Spark and Hadoop

This search system was developed by building an index around the provided BBC Sport document
data. The system was developed using Hortonworks HDP with HDFS as a persistence layer, Spark as the
processing layer, and Livy + Jupyter as the interface layer. The system allows a user to choose which of
the directories available in the dataset to index and then using Livy+Jupyter allows them to query and
review results. The system performs quite well at searching once the data has been loaded. Queries on
average take under a second.

TF-IDF was the primary vectorization technique used. Further details can be found in the PDF documentation.
