// Databricks notebook source
// MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/csv/ggplot2

// COMMAND ----------

// MAGIC %sql
// MAGIC DROP TABLE IF EXISTS movies;
// MAGIC CREATE TABLE movies
// MAGIC USING csv
// MAGIC OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv", header "true")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from movies limit 3

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC movies=spark.read.csv('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv',header=True,inferSchema=True)
// MAGIC display(movies.filter(movies.year>1980))

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC new_movies=movies.dropna().withColumn('rate/budget',movies.rating*1e9/movies.budget).select('title','year','rate/budget').filter(movies.year>1990).groupby('year').agg({'rate/budget':'max'}).sort('year',ascending=1)
// MAGIC display(new_movies)
// MAGIC # new_movies.write.saveAsTable("newMovies")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select title, year, rating*1e9/budget as quality from movies a
// MAGIC where a.rating*1e9/budget>=(select max(rating*1e9/budget) from movies b where a.year=b.year)
// MAGIC and a.year>1980
// MAGIC order by quality desc

// COMMAND ----------

// MAGIC %python
// MAGIC display(movies)

// COMMAND ----------

// MAGIC %python
// MAGIC movies_ml=movies.dropna().sample(fraction=0.2,seed=69)
// MAGIC movies_ml.cache()
// MAGIC 
// MAGIC 
// MAGIC display(movies_ml)
// MAGIC # movies.count()

// COMMAND ----------

// MAGIC %python
// MAGIC movies_ml.printSchema()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.ml.feature import VectorAssembler
// MAGIC from pyspark.ml.regression import RandomForestRegressor,LinearRegression
// MAGIC 
// MAGIC vectorAssembler=VectorAssembler(inputCols=['r'+str(i) for i in range(1,11)],outputCol='features')
// MAGIC data_labeled=vectorAssembler.transform(movies_ml)
// MAGIC data_labeled=data_labeled.select('title','year','features','rating')
// MAGIC 
// MAGIC 
// MAGIC # data_rdd=movies_ml.select('budget','rating').rdd.map(lambda row: [e for e in row])
// MAGIC # data_labeled=data_rdd.map(lambda row: LabeledPoint(row[-1],row[:-1]))
// MAGIC train,test=data_labeled.randomSplit([0.7,0.3])
// MAGIC # display(train)
// MAGIC 
// MAGIC # lr=LinearRegression(featuresCol='features',labelCol='rating',regParam=0.5)
// MAGIC lr=RandomForestRegressor(labelCol='rating')
// MAGIC modelA=lr.fit(train)
// MAGIC predictA_train=modelA.transform(train)
// MAGIC predictA_test=modelA.transform(test)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.evaluation import RegressionEvaluator
// MAGIC evaluator=RegressionEvaluator(metricName='r2',labelCol='rating')
// MAGIC 
// MAGIC RMSE_train=evaluator.evaluate(predictA_train)
// MAGIC RMSE_test=evaluator.evaluate(predictA_test)
// MAGIC 
// MAGIC print('R2_train={:.3f},R2_test={:.3f}'.format(RMSE_train,RMSE_test))

// COMMAND ----------

// MAGIC %python
// MAGIC display(predictA_train)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC from pyspark.sql.functions import udf, col
// MAGIC from pyspark.sql.types import ArrayType, DoubleType
// MAGIC 
// MAGIC def to_array(col):
// MAGIC     def to_array_(v):
// MAGIC         return v.toArray().tolist()
// MAGIC     return udf(to_array_, ArrayType(DoubleType()))(col)
// MAGIC 
// MAGIC # (df
// MAGIC #     .withColumn("xs", to_array(col("vector")))
// MAGIC #     .select(["word"] + [col("xs")[i] for i in range(3)]))
// MAGIC # display(predictA_test)
// MAGIC predictA_test_table=predictA_test.withColumn('rs',to_array(col('features')))
// MAGIC 
// MAGIC predictA_test_table.write.saveAsTable('predict_test',mode='overwrite')

// COMMAND ----------

// MAGIC %python
// MAGIC movies_ml.write.saveAsTable('movies_ml',mode='overwrite')

// COMMAND ----------

// MAGIC %sql
// MAGIC select year, avg(prediction-rating) as delta from predict_test
// MAGIC group by year
// MAGIC order by year 

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.Pipeline

val movie_scala = spark.sql("select * from movies_ml")
// movie_scala.show()
val Array(train, test) = movie_scala.randomSplit(Array(0.7,0.3),seed=69)

val assembler = new VectorAssembler()
  .setInputCols(Array("r1","r2","r3","r4","r5","r6","r7","r8","r9","r10"))
  .setOutputCol("features")

// val data_ml = assembler
//   .transform(movie_scala)
//   .select("title","year","features","rating")

val lr = new RandomForestRegressor()
  .setLabelCol("rating")
  .setFeaturesCol("features")

val pipeline = new Pipeline()
  .setStages(Array(assembler, lr))

val model = pipeline.fit(train)
val predict_train = model.transform(train)
val predict_test = model.transform(test)


// predict_test.show()

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator()
  .setLabelCol("rating")
  .setMetricName("r2")

val r2_train = evaluator.evaluate(predict_train)
val r2_test = evaluator.evaluate(predict_test)

println("R2_train = " + r2_train)
println("R2_test = " + r2_test)

// COMMAND ----------

predict_test.write.mode("overwrite").saveAsTable("predict_test_scala")

// COMMAND ----------

// MAGIC %sql
// MAGIC select a.title, a.year, a.prediction as python, b.prediction as scala, a.prediction-b.prediction as delta from
// MAGIC predict_test a join predict_test_scala b
// MAGIC on a.title=b.title

// COMMAND ----------


