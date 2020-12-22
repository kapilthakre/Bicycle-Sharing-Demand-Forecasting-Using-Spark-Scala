// Databricks notebook source
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.log4j._
import org.apache.spark.sql.functions.to_timestamp
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.RandomForestRegressor

// COMMAND ----------

// MAGIC %md
// MAGIC ##Data Exploration and Transformation

// COMMAND ----------

// MAGIC %md
// MAGIC ##1. Read dataset in Spark

// COMMAND ----------

val trainDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load("/FileStore/tables/edureka/train.csv")
trainDF.show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ##2.Get summary of data and variable types

// COMMAND ----------

trainDF.printSchema

// COMMAND ----------

display(trainDF.describe())

// COMMAND ----------

// MAGIC %md
// MAGIC ##3.Decide which columns should be categorical and then convert them accordingly

// COMMAND ----------

//Cheking unique value In each column
val exprs = trainDF.schema.fields.filter(x => x.dataType != StringType).map(x=>x.name ->"approx_count_distinct").toMap
//data.agg(exprs).show(false)

// COMMAND ----------

display(trainDF.agg(exprs))

// COMMAND ----------

//so we are considering "workingday,holiday,season, and wether column" as a categorical column and we are applying onehotencoder on column with values > 2
val indexer = Array("season","weather").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
val pipeline = new Pipeline().setStages(indexer)
val df_r = pipeline.fit(trainDF).transform(trainDF).drop("season","weather")

// COMMAND ----------

df_r.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ##4.Check for any missing value in data set and treat it

// COMMAND ----------

trainDF.select(trainDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

// COMMAND ----------

// MAGIC %md
// MAGIC ##5.Explode season column into separate columns such as season_<val>and drop season
// MAGIC ##6.Execute the same for weather as weather_<val> and drop weather

// COMMAND ----------

//Ans : We don’t need to explode season column and weather column because previously I already handle categorical column with values > 2 by applying onehotencoder.

// COMMAND ----------

// MAGIC %md
// MAGIC ##7. Split datetime in to meaning columns such as hour, day, month, year, etc.

// COMMAND ----------

//Converting datetime string column to timestamp column
val df_time = df_r.withColumn("datetime", to_timestamp(col("datetime"),"d-M-y H:m"))

//Now Spliting date time into meaning columns such as year,month,day,hour
val datetime_trainDF = df_time.
withColumn("year", year(col("datetime"))).
withColumn("month", month(col("datetime"))).
withColumn("day", dayofmonth(col("datetime"))).
withColumn("hour", hour(col("datetime"))).
withColumn("minute",minute(col("datetime")))

// COMMAND ----------

// MAGIC %md
// MAGIC ##8.Explore how count varies with different features such as hour,month,etc

// COMMAND ----------

datetime_trainDF.groupBy("year").count.show()
datetime_trainDF.groupBy("month").count.show()
datetime_trainDF.groupBy("day").count.show()
datetime_trainDF.groupBy("hour").count.show()
datetime_trainDF.groupBy("minute").count.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ##Model Development

// COMMAND ----------

// MAGIC %md
// MAGIC ##1.Split the data set into train and train_test

// COMMAND ----------

val splitSeed = 123
val Array(train,train_test) = datetime_trainDF.randomSplit(Array(0.7,0.3),splitSeed)

// COMMAND ----------

// MAGIC %md
// MAGIC ##2. Try different regression algorithms such as linear regression, random forest, etc. and note accuracy

// COMMAND ----------

//Generate Feature Column
val feature = Array("holiday","workingday","temp","atemp","humidity","windspeed","season_Vec","weather_Vec","year","month","day","hour","minute")
//Assemble Feature Column
val assembler = new VectorAssembler().setInputCols(feature).setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ##Linear Regression Model

// COMMAND ----------

//Model Building
val lr = new LinearRegression().setLabelCol("count").setFeaturesCol("features")

//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,lr))

//Training Model
val lrModel = pipeline.fit(train)
val predictions = lrModel.transform(train_test)

//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Linear Regression Root Mean Squared Error (RMSE) on train_test data = " + rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC ##GBT Regressor

// COMMAND ----------

//Model Building
val gbt = new GBTRegressor().setLabelCol("count").setFeaturesCol("features")

//Creating pipeline
val pipeline = new Pipeline().setStages(Array(assembler,gbt))

//Training Model
val gbtModel = pipeline.fit(train)
val predictions = gbtModel.transform(train_test)

//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("GBT Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC ##Decision Tree Regressor

// COMMAND ----------

//Model Building
val dt = new DecisionTreeRegressor().setLabelCol("count").setFeaturesCol("features")

//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,dt))

//Training Model
val dtModel = pipeline.fit(train)
val predictions = dtModel.transform(train_test)

//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Decision Tree Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC ##Random Forest Regressor

// COMMAND ----------

//Model Building
val rf = new RandomForestRegressor().setLabelCol("count").setFeaturesCol("features")

//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,rf))

//Training Model
val rfModel = pipeline.fit(train)
val predictions = rfModel.transform(train_test)

//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Random Forest Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC ##3.Select the best model and persistit

// COMMAND ----------

//So as we try diferent Regression Alorithms and found that "GBT Regressor Model" is giving better accuracy compare to other.
//gbtModel.write.overwrite().save("/FileStore/tables/model/bicycle-model")

// COMMAND ----------

// MAGIC %md
// MAGIC ##Model Implementation and Prediction
// MAGIC Application Development for Model Generation
// MAGIC 
// MAGIC 1. Clean and Transform the data
// MAGIC 
// MAGIC 2. Develop the model and persist it.

// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoder

object BicyclePredict{
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("kapil")
    val sc = new SparkContext(sparkConf)
    
    sc.setLogLevel("ERROR")

    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    
    println("Reading training data...................")
    
    val trainDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load("/FileStore/tables/edureka/train.csv")
    
    println("Cleaning data.................")
    
    //Converting datetime string column to timestamp column
    val df_time = trainDF.withColumn("datetime", to_timestamp(col("datetime"),"d-M-y H:m"))

    //Now Spliting date time into meaning columns such as year,month,day,hour
    val datetime_trainDF = df_time.
    withColumn("year", year(col("datetime"))).
    withColumn("month", month(col("datetime"))).
    withColumn("day", dayofmonth(col("datetime"))).
    withColumn("hour", hour(col("datetime"))).
    withColumn("minute",minute(col("datetime")))   
    
    //Onehot encoding on season and weather column.
    val indexer = Array("season","weather").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
    val pipeline = new Pipeline().setStages(indexer)
    val df_r = pipeline.fit(datetime_trainDF).transform(datetime_trainDF)
    
    //split data into train test
    val splitSeed =123
    val Array(train, train_test) = df_r.randomSplit(Array(0.7, 0.3), splitSeed)
    
    //Generate Feature Column
    val feature_cols = Array("holiday","workingday","temp","atemp","humidity","windspeed","season_Vec","weather_Vec","year","month","day","hour","minute")
    
    //Assemble Feature
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
    
    //Model Building
    val gbt = new GBTRegressor().setLabelCol("count").setFeaturesCol("features")
    
    val pipeline2 = new Pipeline().setStages(Array(assembler,gbt))
    
    println("Training model................")
    val gbt_model = pipeline2.fit(train)
    val predictions = gbt_model.transform(train_test)
    
    val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("GBT Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)
  
    println("Persisting the model................")
    gbt_model.write.overwrite().save("/FileStore/tables/model/bicycle-model")
  }
}

// COMMAND ----------

//Application Execution
spark2-submit --class "BicyclePredict" --master yarn /mnt/home/edureka_836462/BicycleProject/BicycleTrain/target/scala-2.11/bicycletrain_2.11-1.0.jar

// COMMAND ----------

// MAGIC %md
// MAGIC ##Application Development for Demand Prediction
// MAGIC 
// MAGIC Model Prediction Application – Write an application to predict the bike demand based on the input dataset from HDFS:
// MAGIC 
// MAGIC 1. Load the persisted model.
// MAGIC 
// MAGIC 2. Predict bike demand
// MAGIC 
// MAGIC 3. Persist the result to RDBMS

// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoder

object BicyclePredict {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Telecom")
    val sc = new SparkContext(sparkConf)

    sc.setLogLevel("ERROR")

    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    
    println("Reading Training data.................")
    
    val testDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load("/FileStore/tables/edureka/test.csv")
    
    println("Cleaning data.................")
    
    //Converting datetime string column to timestamp column
    val df_time = testDF.withColumn("datetime", to_timestamp(col("datetime"),"d-M-y H:m"))
    
    //Now Spliting date time into meaning columns such as year,month,day,hour
    val datetime_testDF = df_time.
    withColumn("year", year(col("datetime"))).
    withColumn("month", month(col("datetime"))).
    withColumn("day", dayofmonth(col("datetime"))).
    withColumn("hour", hour(col("datetime"))).
    withColumn("minute",minute(col("datetime")))
    
    //Onehot encoding on season and weather column.
    val indexer = Array("season","weather").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
    val pipeline = new Pipeline().setStages(indexer)
    val df_r = pipeline.fit(datetime_testDF).transform(datetime_testDF)
    
    println("Loading Trained Model..............")
    val gbt_model = PipelineModel.read.load("/FileStore/tables/model/bicycle-model")
    
    println("Making predictions...........") 
    val predictions = gbt_model.transform(df_r).select($"datetime",$"prediction".as("count"))
    
    println("Persisting the result to RDBMS..............")
    predictions.write.format("jdbc").
      option("url", "jdbc:mysql://mysqldb.edu.cloudlab.com/kapil_bicycle").
      option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "predictions").
      option("user", "labuser").
      option("password", "edureka").
      mode(SaveMode.Append).save
  }
}

// COMMAND ----------

// MAGIC %md
// MAGIC ##Application for Streaming Data
// MAGIC 
// MAGIC Write an application to predict demand on streaming data:
// MAGIC 
// MAGIC ##1. Setup flume to push data into spark flume sink.

// COMMAND ----------

//Kafka topic creation:
kafka-topics --create --zookeeper ip-20-0-21-161.ec2.internal:2181 --replication-factor 1 --partitions 1 --topic edureka_836462_bicycle_kapil

// COMMAND ----------

// MAGIC %md
// MAGIC ##Flume configuration:

// COMMAND ----------

agent1.sources  = source1
agent1.channels = channel1
agent1.sinks = spark
agent1.sources.source1.type = org.apache.flume.source.kafka.KafkaSource
agent1.sources.source1.kafka.bootstrap.servers = ip-20-0-31-210.ec2.internal:9092
agent1.sources.source1.kafka.topics = edureka_836462_bicycle_kapil
agent1.sources.source1.kafka.consumer.group.id = edureka_836462_bicycle_kapil
agent1.sources.source1.channels = channel1
agent1.sources.source1.interceptors = i1
agent1.sources.source1.interceptors.i1.type = timestamp
agent1.sources.source1.kafka.consumer.timeout.ms = 100
agent1.channels.channel1.type = memory
agent1.channels.channel1.capacity = 10000
agent1.channels.channel1.transactionCapacity = 1000
agent1.sinks.spark.type = org.apache.spark.streaming.flume.sink.SparkSink
agent1.sinks.spark.hostname = ip-20-0-41-62.ec2.internal
agent1.sinks.spark.port = 4143
agent1.sinks.spark.channel = channel1

// COMMAND ----------

// MAGIC %md
// MAGIC ## Run Flume agent:

// COMMAND ----------

flume-ng agent --conf conf --conf-file bicycle.conf --name agent1 -Dflume.root.logger=DEBUG,console

// COMMAND ----------

// MAGIC %md
// MAGIC ##2. Configure spark streaming to pulldata from spark flume sink using receivers and predict the demand using model and persist the result to RDBMS.

// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.flume._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoder

object BicycleStreaming {
  case class Bicycle(datetime: String, season: Int, holiday: Int, workingday: Int, weather: Int, temp: Double, atemp: Double, humidity: Int, windspeed: Double)

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("kapil")
    val sc = new SparkContext(sparkConf)
    val ssc = new StreamingContext(sc, Seconds(2))

    sc.setLogLevel("ERROR")

    val spark = new org.apache.spark.sql.SQLContext(sc)

    import spark.implicits._

    val flumeStream = FlumeUtils.createPollingStream(ssc, "ip-20-0-41-62.ec2.internal", 4143)

    println("Loading tained model.............")    
    val gbt_model = PipelineModel.read.load("/user/edureka_836462/bicycle-model")

    
    val lines = flumeStream.map(event => new String(event.event.getBody().array(), "UTF-8"))

    lines.foreachRDD { rdd => 
      def row(line: List[String]): Bicycle = Bicycle(line(0), line(1).toInt, line(2).toInt,
              line(3).toInt, line(4).toInt, line(5).toDouble, line(6).toDouble, line(7).toInt,
              line(8).toDouble
              )

      val rows_rdd = rdd.map(_.split(",").to[List]).map(row)
      val rows_df = rows_rdd.toDF
    
      if(rows_df.count > 0) {
        
        val df_time = rows_df.withColumn("datetime",to_timestamp(col("datetime"),"d-M-y H:m"))
        val datetime_testDF = df_time.
        withColumn("year", year(col("datetime"))).
        withColumn("month", month(col("datetime"))).
        withColumn("day", dayofmonth(col("datetime"))).
        withColumn("hour", hour(col("datetime"))).
        withColumn("minute",minute(col("datetime")))

        //Onehot encoding on season nd weather column.
        val indexer = Array("season","weather").map(c => new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
        val pipeline = new Pipeline().setStages(indexer)
        val df_r = pipeline.fit(datetime_testDF).transform(datetime_testDF)

        println("Making predictions...............")   
        val predictions =  gbt_model.transform(df_r).select($"datetime",$"prediction".as("count"))

        println("Persisting the result to RDBMS..................")
        predictions.write.format("jdbc").
          option("url", "jdbc:mysql://mysqldb.edu.cloudlab.com/kapil64_bicycle").
          option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "predictions").
          option("user", "labuser").
          option("password", "edureka").
          mode(SaveMode.Append).save
      }
    }
    
    ssc.start()
    ssc.awaitTermination()    
  }
}

// COMMAND ----------

// MAGIC %md
// MAGIC ##Run the application:

// COMMAND ----------

spark2-submit --packages mysql:mysql-connector-java:8.0.13 --class "BicycleStreaming" --master yarn /mnt/home/edureka_836462/BicycleProject/BicycleStreaming/target/scala-2.11/bicyclestreaming_2.11-1.0.jar

// COMMAND ----------

// MAGIC %md
// MAGIC ##3. Push messages from flume to test the application. Here application should process and persist the result to RDBMS

// COMMAND ----------

kafka-console-producer --broker-list ip-20-0-31-210.ec2.internal:9092 --topic edureka_836462_bicycle_kapil

// COMMAND ----------

1/20/2011 0:00,1,0,1,1,10.66,11.365,56,26.0027
