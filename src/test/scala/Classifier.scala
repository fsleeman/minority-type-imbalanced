package edu.vcu.sleeman

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame

import scala.util.Random
import org.apache.log4j._
import org.apache.spark.ml.param.Param
import org.apache.spark.mllib.util.MLUtils

import scala.collection.mutable
import scala.reflect.ClassTag


//FIXME - turn classes back to Ints instead of Doubles

object Classifier {

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)


  def getIndexedDF(spark: SparkSession, columns: Array[String], trainData: DataFrame): DataFrame = {
    val keepCols = trainData.columns.map(status => if (columns.contains(status) && !status.equals("_c41")) None else status)
    val indexers = columns.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_idx")
    }
    val pipeline = new Pipeline()
      .setStages(indexers)

    val newDF = pipeline.fit(trainData).transform(trainData)
    val filteredDF = newDF.select(newDF.columns.filter(colName => !columns.contains(colName)).map(colName => new Column(colName)): _*)
    filteredDF
  }


  def runClassifier(spark: SparkSession, data: DataFrame, samplingMethod: String) {
    import spark.implicits._

    val maxLabels = data.select("label").distinct().count()
    val numSamples = 1000
    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2)) //FIXME - does this need to be stratified?
    println("train data size: " + trainData.count())
    // Sample data
    val filteredDF2 = samplingMethod match {
      case "undersample" => underSample(spark, trainData, numSamples)
      case "oversample" => overSample(spark, trainData, numSamples)
      case "smote" => smote(spark, trainData, numSamples)
      case _ => trainData
    }

    val maxLabel: Int = testData.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).max
    println("train data size after: " + filteredDF2.count())
    println("test data size: " + testData.count())

    println("Sampled Counts")
    val aggregatedCounts = filteredDF2.groupBy("label").agg(count("label")) //FIXME
    aggregatedCounts.show()

    testData.show()
    println("test counts")
    getCountsByClass(spark, "label", testData).sort("_2")

    println("train counts")
    getCountsByClass(spark, "label", filteredDF2).sort("_2")

    val inputCols = data.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    println("total sampled training size: " + filteredDF2.count())
    val assembledTrainData = assembler.transform(filteredDF2)
    val assembledTestData = assembler.transform(testData)

    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(Random.nextLong()).
      setLabelCol("label").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val model = classifier.fit(assembledTrainData)
    val predictions = model.transform(assembledTestData)
    val testLabels = testData.select("label").distinct().map(_.getAs[Double]("label")).map(x => x.toInt).collect().sorted

    val confusionMatrix = predictions.
      groupBy("label").
      pivot("prediction", (0 to maxLabel)).
      count().
      na.fill(0.0).
      orderBy("label")

    //predictions.show()
    // precision=TP / (TP + FP) 
    //sensitivity = TP / (TP + FN)
    //specificity = TN / (FP + TN) 
    //F-score = 2*TP /(2*TP + FP + FN) 

    confusionMatrix.show()

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum

    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))
    classMaps.foreach(println)

    var sensitiviySum = 0.0
    var sensitiviyCount = 0
    //FIXME - could be made parallel w/udf
    for (clsIndex <- 0 to maxLabel) {
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0
      val tp = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn = colSum - tp
      val fp = rowValueSum - tp
      val tn = totalCount - tp - fp - fn

      println("tp: " + tp + " fp: " + fp + " tn: " + tn + " fn: " + fn)
      val sensitivity = tp / (tp + fn).toFloat
      if (tp + fn > 0) {
        sensitiviySum += sensitivity
        sensitiviyCount += 1
      }

    }
    println(sensitiviyCount + " " + sensitiviySum)
    println("AvAcc: " + sensitiviySum / sensitiviyCount)
  }

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount: Int): String = {
    if (kCount >= 4) {
      return "safe"
    }
    else if (kCount >= 2) {
      return "borderline"
    }
    else if (kCount == 1) {
      return "rare"
    }
    else {
      return "outlier"
    }
  }

  def setMinorityStatus(cls: Int, sample: (Int, Array[Int])): (Int, String) = {
    //positive case
    if (cls == sample._1) {
      val matchingCount = sample._2.filter(x => x == cls).length
      (1, getMinorityClassLabel(matchingCount))
    }
    //negative case
    else {
      val matchingCount = sample._2.filter(x => x != cls).length
      (0, getMinorityClassLabel(matchingCount))
    }
  }

  def getMinorityClassResults(cls: Int, name: String, data: Array[NearestClassResult]) {
    val minorityStatus = data.map(x => setMinorityStatus(cls, x))

    val positiveSamples = minorityStatus.filter(x => x._1 == 1).map(x => x._2)
    val negativeSamples = minorityStatus.filter(x => x._1 != 1).map(x => x._2)

    val positiveSafeCount = positiveSamples.filter(x => (x == "safe")).length
    val positiveBorderlineCount = positiveSamples.filter(x => (x == "borderline")).length
    val positiveRareCount = positiveSamples.filter(x => (x == "rare")).length
    val positiveOutlierCount = positiveSamples.filter(x => (x == "outlier")).length
    val positiveResults = "Positive: " + positiveSamples.length + "\nsafe: " + positiveSafeCount + " borderline: " + positiveBorderlineCount + " rare: " + positiveRareCount + " outlier: " + positiveOutlierCount

    val negativeSafeCount = negativeSamples.filter(x => (x == "safe")).length
    val negativeBorderlineCount = negativeSamples.filter(x => (x == "borderline")).length
    val negativeRareCount = negativeSamples.filter(x => (x == "rare")).length
    val negativeOutlierCount = negativeSamples.filter(x => (x == "outlier")).length
    val negativeResults = "Negative: " + negativeSamples.length + "\nsafe: " + negativeSafeCount + " borderline: " + negativeBorderlineCount + " rare: " + negativeRareCount + " outlier: " + negativeOutlierCount

    println("\nClass " + name + "\n" + positiveResults + "\n" + negativeResults + "\n")
  }

  def getDistances2(current: Element, train: Array[Element]): NearestClassResultIndex = {
    val result = train.map(x => getDistanceValue(x, current)).sortBy(x => x._1).take(5)
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    val cls = current._2._1
    //val sum = result.filter(x=>(x._2==cls)).length
    return (current._1, current._2._1, result.map(x => x._2))
  }

  def getDistanceValue(train: Element, test: Element): DistanceResult = {
    if (train._1 == test._1) {
      return (Float.MaxValue, train._2._1)
    }
    else {
      val zipped = test._2._2.zip(train._2._2)
      val result = zipped.map({ case (x, y) => (x - y) * (x - y) })
      return ((result.sum), train._2._1) //removed sqrt
    }
  }

  def mapRow(currentRow: Array[Any]) = {
    val reverseRow = currentRow.reverse
    val cls = reverseRow.head.toString().toFloat.toInt
    val features = reverseRow.tail.map(_.toString().toFloat)
    (cls, features)
  }

  def calculateMinorityClasses(spark: SparkSession, trainData: DataFrame) {
    trainData.show()

    val trainRDD = trainData.rdd.map(_.toSeq.toArray).map(x => mapRow(x))
    trainRDD.count()

    //FIXME - is this better with broadcasting?
    val train_index = trainRDD.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    println("******************** Class Stats *****************************")
    val countOfClasses = trainRDD.map((_, 1L)).reduceByKey(_ + _).map { case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey
    val countResults = countOfClasses.map(x => (x._1, x._2.count(x => true)))
    countResults.sortBy(x => x._1, true).collect().foreach(println);

    val classes = countResults.sortBy(x => x._1).map(x => x._1)
    classes.foreach(println)

    println("**************************************************************")

    println("***************** Minority Class Over/Under Resample ****************************")
    val t0 = System.nanoTime()

    val train_data_collected = train_index.collect()
    val tX = System.nanoTime()
    println("Time 1: " + (tX - t0) / 1.0e9 + "s")

    val minorityData = train_index.map(x => getDistances2(x, train_data_collected)).cache()


    val minorityDataCollected = minorityData.collect()
    val indexedLabelNames = getIndexedLabelNames(trainData)
    val rows: Array[Row] = indexedLabelNames.collect

    for (cls <- classes) {
      val res = rows.filter(x => x(0) == cls)
      println()
      getMinorityClassResults(cls, res(0)(1).toString(), minorityDataCollected.map(x => (x._2, x._3)))
    }
  }

  def convertIndexedToName(cls: Int, indexedLabelNames: DataFrame): String = {
    val rows: Array[Row] = indexedLabelNames.collect
    val res = rows.filter(x => x(0) == cls)
    return res(0)(1).toString()
  }

  def getIndexedLabelNames(df: DataFrame): DataFrame = {
    val converter = new IndexToString()
      .setInputCol("label")
      .setOutputCol("originalCategory")
    val converted = converter.transform(df)
    converted.select("label", "originalCategory").distinct()
  }

  //assume there is only one class present
  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    println("Total Count: " + df.count())
    //val numLabels = df.select("label").distinct().count().toInt
    //println("num labels: " + numLabels)
    df.select("label").distinct().show()
    //val labelCounts = df.groupBy("label").agg(count("label")).take(numLabels)
    //val maxLabelCount = labelCounts.map(x => x(1).toString().toInt).reduceLeft(_ max _)
    //println("\tmax count: " + maxLabelCount)

    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    //for (l <- 0 to numLabels) {
      //val currentCase = df.filter(df("label") === l)
      val samplesToAdd = numSamples - df.count()
      println("\t\tto add: " + samplesToAdd)
      val currentCount = df.count()
      if (0 < currentCount && currentCount < numSamples) {
        val currentSamples = df.sample(true, (numSamples - currentCount) / currentCount.toDouble).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples
      }

    /*if (df.count() == 0) {
        val currentSamples = currentCase.sample(true, (numSamples)).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples

      }
      else {
        val currentSamples = currentCase.sample(true, (numSamples - currentCase.count()) / currentCase.count().toDouble).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples

      }*/
      //FIXME - make this faster

      //FIXME - add original samples
      //samples ++ currentCase.sample(true, (maxLabelCount - currentCase.count()/currentCase.count().toDouble)).collect()
      //val totalResults = df.union(currentCase.sample(true, (maxLabelCount - currentCase.count()/currentCase.count().toDouble)))
      //println(totalResults.count())
   //// }
    println("new count: " + samples.length)

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    val xxx = df.union(x).toDF()
    println("joined count: " + xxx.count())
    return xxx//df.union(x).toDF()
  }

  def underSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    println("~~~~~~~~~~~~~~~~~ Under Sample")
    //val counts = getCountsByClass(spark, "label", df).collect().map(x => x(1).toString().toInt).sorted
    //counts.foreach(println)

    //val undersampleCount = counts(counts.length / 2).toInt
    var samples = Array[Row]() //FIXME - make this more parallel

    //for (cls <- 0 to counts.length) {
      //val currentClass = df.filter(df("label") === cls)
      val underSampleRatio = numSamples / df.count().toDouble
      if (underSampleRatio < 1.0) {
        val currentSamples = df.sample(false, underSampleRatio).collect()
        samples = samples ++ currentSamples
        println("new samples: " + samples.length) //FIXME - not working correctly
        val foo = spark.sparkContext.parallelize(samples)
        val x = spark.sqlContext.createDataFrame(foo, df.schema)
        return x
      }
      else {
        println("new samples: " + df.count())
        //samples = samples ++ df.collect()
        return df
      }
    //}


  }


  def smote(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    val numClasses = df.select(df("label")).distinct().count().toInt
    val aggregatedCounts = df.groupBy("label").agg(count("label"))
    println("SMOTE counts")
    aggregatedCounts.show()
    println(numClasses + " " + aggregatedCounts.count())
    val maxClassCount = aggregatedCounts.select("count(label)").collect().toSeq.map(x => x(0).toString.toInt).max
    println(maxClassCount)
    //val smoteTo = maxClassCount / 2
    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    for (cls <- 0 to numClasses) {
      print("cls: " + cls + " ")
      val cnt = aggregatedCounts.filter(aggregatedCounts("label") === cls.toDouble).collect()
      print(cnt.length)
      if (cnt.length == 1) { // make sure this cls exist in training set

        println(cnt(0)(0), cnt(0)(1))
        val currentCount = cnt(0)(1).toString().toInt

        val currentClass = df.filter(df("label") === cls)
        if (currentCount < numSamples) {
          val samplesToAdd = numSamples - currentCount
          println(cls + " adding " + samplesToAdd)

          val currentClassZipped = currentClass.collect().zipWithIndex

          var smoteSamples = ArrayBuffer[Row]()
          for (s <- 1 to samplesToAdd.toInt) {
            def r = scala.util.Random.nextInt(currentClassZipped.length)

            val rand = Array(r, r, r, r, r)
            val sampled = currentClassZipped.filter(x => (rand.contains(x._2))).map(x => x._1) //FIXME - issues not taking duplicates

            val xxxxx = (sampled.toList.map(x => x.toSeq.toList.map(_.toString().toDouble)))
            val ddd = xxxxx.toList.transpose.map(_.sum / xxxxx.length)
            val r2 = Row.fromSeq(ddd.toSeq)

            smoteSamples += r2
          }

          samples = samples ++ smoteSamples
        }
        else {
          //
        }
        samples = samples ++ currentClass.collect()
      }
    }
    println(samples(0))

    println("Number of added SMOTE samples: " + samples.length)
    val rdd = spark.sparkContext.makeRDD(samples)


    val xxx = df.schema.map(x => StructField(x.name, DoubleType, true))
    val smoteDF = spark.createDataFrame(rdd, StructType(xxx))

    val finalDF = underSample(spark, smoteDF, numSamples) //FITME - check if this is the right number

    println("New total count: " + smoteDF.count())
    println("Final total count: " + finalDF.count())

    return finalDF
  }

  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame = {
    import spark.implicits._

    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(100) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString(), x(1).toString().toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    rdd.toDF()
  }

  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect()

    val overSampleCount = 1000
    val underSampleCount = 250
    val smoteSampleCount = 1000
    var dfs: Array[DataFrame] = Array[DataFrame]()
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, overSampleCount)
        case "oversample" => overSample(spark, currentCase, underSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount)
        case _ => currentCase
      }
      dfs = dfs :+ filteredDF2
    }

    println(dfs(0).count())
    println(dfs(1).count())
    println(dfs(2).count())
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val all = dfs.reduce(_ union  _)

    println("ALL count: " + all.count())
    return all

  }


  def minorityTypeResample(spark: SparkSession, df: DataFrame, minorityTypes: Array[String]): DataFrame = {
    val numLabels = df.select("label").distinct().count().toInt

    println("num labels: " + numLabels)
    val maxLabelCount = 200


    var samples = Array[Row]() //FIXME - make this more parallel

    //val types =  Array("safe", "borderline")//Array("safe", "borderline")
    //FIXME - some could be zero if split is too small
    var pickedTypes = df.filter(x => (minorityTypes contains (x(2))))

    val d = df.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect()

    //FIXME - avoid passing spark as parameter?
    val combinedDf = sampleData(spark, pickedTypes, "undersample")

    println("pickedType count: " + pickedTypes.count())
    println("----------- total sampled count: " + combinedDf.count())
    /*for (l <- presentClasses) {
      val samplingMethod = "oversample"

      val currentCase = pickedTypes.filter(pickedTypes("label") === l)
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase)
        case "oversample" => overSample(spark, currentCase)
        case "smote" => smote(spark, currentCase)
        case _ => currentCase
      }



      /*println(l)
      val currentCase = pickedTypes.filter(pickedTypes("label") === l)
      if (currentCase.count() == 0) {
        val currentSamples = currentCase.sample(true, (maxLabelCount)).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples
      }
      else {
        println(maxLabelCount + " " + currentCase.count())
        if(maxLabelCount - currentCase.count()>0) {
          val currentSamples = currentCase.sample(true, (maxLabelCount - currentCase.count()) / currentCase.count().toDouble).collect()
          println("samples created: " + currentSamples.length)
          samples = samples ++ currentSamples
        }
      }*/
      //FIXME - make this faster
    }*/
    /*println("total created samples: " + samples.length)

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, pickedTypes.schema)

    val combinedDf = pickedTypes.union(x).toDF()
*/



    convertFeaturesToVector(combinedDf)
  }

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }

  def runClassifierMinorityType(train: DataFrame, test: DataFrame) {
    val spark = train.sparkSession
    import spark.implicits._
    //train.show()
    //val maxLabels = train.select("label").distinct().count()
    val maxLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).max

    println("test counts")
    getCountsByClass(spark, "label", test).show()

    println("train counts")
    getCountsByClass(spark, "label", train).show()

    test.show()

    val inputCols = test.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val assembledTestData = assembler.transform(test)

    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(Random.nextLong()).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")


    val model = classifier.fit(train)
    val predictions = model.transform(assembledTestData)
    val testLabels = test.select("label").distinct().map(_.getAs[Double]("label")).map(x => x.toInt).collect().sorted

    val confusionMatrix = predictions.
      groupBy("label").
      pivot("prediction", (0 to maxLabel)).
      count().
      na.fill(0.0).
      orderBy("label")

    //predictions.show()
    // precision=TP / (TP + FP)
    //sensitivity = TP / (TP + FN)
    //specificity = TN / (FP + TN)
    //F-score = 2*TP /(2*TP + FP + FN)

    confusionMatrix.show()
    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum

    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))
    classMaps.foreach(println)

    var sensitiviySum = 0.0
    var sensitiviyCount = 0
    //FIXME - could be made parallel w/udf
    for (clsIndex <- 0 to maxLabel) {
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0
      val tp = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn = colSum - tp
      val fp = rowValueSum - tp
      val tn = totalCount - tp - fp - fn

      println("tp: " + tp + " fp: " + fp + " tn: " + tn + " fn: " + fn)
      val sensitivity = tp / (tp + fn).toFloat
      if (tp + fn > 0) {
        sensitiviySum += sensitivity
        sensitiviyCount += 1
      }

    }
    println(sensitiviyCount + " " + sensitiviySum)
    println("AvAcc: " + sensitiviySum / sensitiviyCount)
  }

  import edu.vcu.sleeman.MinorityType.getMinorityTypeStatus

  def runSparKNN(df: DataFrame): Unit = {
    println("Sampled Counts")
    val aggregatedCounts = df.groupBy("label").agg(count("label")) //FIXME
    aggregatedCounts.show()


    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2))
    println("Train Data Count: " + trainData.count())
    println("Test Data Count: " + testData.count())

    getCountsByClass(testData.sparkSession,"label", testData).show()

    val minorityDF = getMinorityTypeStatus(trainData)
    minorityDF.show()

    val minorityTypes = Array("safe", "borderline")
    val trainDataResampled = minorityTypeResample(minorityDF.sparkSession, minorityDF, minorityTypes)
    trainDataResampled.show()
    //trainDataResampled.take(1).foreach(println)
    println("trainDataResampled:" + trainDataResampled.count())
    getCountsByClass(trainDataResampled.sparkSession,"label", trainDataResampled).show()

    runClassifierMinorityType(trainDataResampled, testData)
  }

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    val input_file = args(0)
    val labelColumnName = args(1)
    val useHeader = if (args.length > 2 && args(2).equals("yes")) true else false

    val df1 = spark.read.
      option("inferSchema", true).
      option("header", useHeader).
      csv(input_file)


    val df = df1.repartition(8)

    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)

    val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)
    runSparKNN(preppedDataUpdated)

     //runClassifier(spark, preppedDataUpdated, "None")
  }
}
  