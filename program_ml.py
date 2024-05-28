from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, regexp_replace
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import udf, count
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("NetworkAttackDetection").getOrCreate()

# Load the synthetic labeled data
df = spark.read.csv('labeled_attacks.csv', header=True, inferSchema=True)

# Convert the 'Time' column to a timestamp
df = df.withColumn('Time', unix_timestamp(col('Time')).cast('timestamp'))

# Extract features and labels (example)
feature_columns = ['Duration', 'SrcPackets', 'DstPackets', 'SrcBytes',
                   'DstBytes', 'SrcPort', 'DstPort']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df = assembler.transform(df)

# Index the labels
indexer = StringIndexer(inputCol='AttackType', outputCol='label')
df = indexer.fit(df).transform(df)

# Select only the features and label columns
df = df.select('features', 'label')

# Split the data into training and test sets (if needed, or choose manually)
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)


# Initialize the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Create a parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Initialize cross-validator
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

# Train the model using cross-validation
cvModel = crossval.fit(train_df)

# Evaluate the model on the test set
predictions = cvModel.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                              predictionCol='prediction',
                                              metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f'Test set accuracy = {accuracy}')

# #########################################################
# Load new data for tests
new_data_spark_df = spark.read.csv('data_for_tests.csv',
                                   header=True,
                                   inferSchema=True)

# Preprocess
new_data_spark_df = new_data_spark_df.withColumn(
    'SrcPort', regexp_replace(col('SrcPort'), 'Port', '').cast('int'))
new_data_spark_df = new_data_spark_df.withColumn(
    'DstPort', regexp_replace(col('DstPort'), 'Port', '').cast('int'))
new_data_spark_df = new_data_spark_df.withColumn(
    'SrcDevice', regexp_replace(col('SrcDevice'), 'Comp', '').cast('int'))
new_data_spark_df = new_data_spark_df.withColumn(
    'DstDevice', regexp_replace(col('DstDevice'), 'Comp', '').cast('int'))

# Check the transformed data
new_data_spark_df.show()

# Define feature columns
feature_columns = ['Duration', 'SrcPackets', 'DstPackets', 'SrcBytes',
                   'DstBytes', 'SrcPort', 'DstPort']

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
new_data_spark_df = assembler.transform(new_data_spark_df)

# Check if the number of columns match
print(f"Feature columns: {feature_columns}")
print(f"Number of features in the model: {len(feature_columns)}")

# Use the trained model to predict attack types
predictions = cvModel.transform(new_data_spark_df)

# Show predictions
predictions.select('features', 'prediction').show()

# Example attacks
attack_mapping = {
    0: 'DNS Amplification',
    1: 'Malware Communication',
    2: 'DoS',
    3: 'Port Scanning',
    4: 'Brute Force',
    5: 'Data Exfiltration',
    6: 'MITM',
    7: 'UDP Flood',
    8: 'SQL Injection',
    9: 'ARP Spoofing',
    10: 'DNS Tunnelling',
    11: 'ICMP Ping DOS',
    12: 'FTP DOS',
    13: 'DNS Rebinding',
    14: 'TCP SYN Flood'
}


def map_prediction_to_attack(prediction):
    return attack_mapping.get(int(prediction), 'Unknown')


map_prediction_udf = udf(map_prediction_to_attack, StringType())

# Apply the UDF to the predictions DataFrame
predictions_with_attacks = predictions.withColumn(
    'attack_type', map_prediction_udf(col('prediction')))

# Count the number of each type of attack
attack_counts = predictions_with_attacks.groupBy(
    'attack_type').agg(count('*').alias('count')).orderBy('count',
                                                          ascending=False)

# Show the attack counts
attack_counts.show()
