package edu.vcu.sleeman

import breeze.linalg.DenseVector
import org.apache.avro.generic.GenericData.StringType
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types._

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
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction

import scala.collection.mutable
import scala.reflect.ClassTag
import org.apache.spark.ml.classification._
import org.apache.spark.ml.knn.KNN

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


  def runClassifier(spark: SparkSession, df: DataFrame, samplingMethod: String) {
    import spark.implicits._
    //df.show()

    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2),42L)


    val train_index = trainData.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val train_data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, "", rowMapped.reverse)
    })


    val results: DataFrame = train_data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minority_type")
      .withColumnRenamed("_4", "features")
    //val sampledData = sampleData(spark, data, samplingMethod)
    //results.show()

    val trainDataSampled = sampleData(spark, results, samplingMethod)
    //getCountsByClass(spark, "label", trainDataSampled)

    val maxLabel: Int = testData.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).max

    println("test counts")
    getCountsByClass(spark, "label", testData).show()

    println("train counts")
    getCountsByClass(spark, "label", trainDataSampled).show()

    println("** train data **")
    trainDataSampled.show()
    trainDataSampled.printSchema()

    //testData.show()

    val inputCols = testData.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val assembledTestData = assembler.transform(testData)

    val classifier = new RandomForestClassifier().setNumTrees(10).
      //setSeed(Random.nextLong()).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

    val convertedDF = trainDataSampled//convertFeaturesToVector(trainDataSampled)

    val model = classifier.fit(convertedDF)
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

//!!!confusionMatrix.show()
     val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
     val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum

     val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))
     //classMaps.foreach(println)

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

       //println("tp: " + tp + " fp: " + fp + " tn: " + tn + " fn: " + fn)
       val sensitivity = tp / (tp + fn).toFloat
       if (tp + fn > 0) {
         sensitiviySum += sensitivity
         sensitiviyCount += 1
       }

     }
     println(sensitiviyCount + " " + sensitiviySum)
     println("AvAcc: " + sensitiviySum / sensitiviyCount)




    //val maxLabels = data.select("label").distinct().count()
    /*val numSamples = 200
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
    println("AvAcc: " + sensitiviySum / sensitiviyCount)*/
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
    //println("Total Count: " + df.count())
    //val numLabels = df.select("label").distinct().count().toInt
    //println("num labels: " + numLabels)
    //df.select("label").distinct().show()
    //val labelCounts = df.groupBy("label").agg(count("label")).take(numLabels)
    //val maxLabelCount = labelCounts.map(x => x(1).toString().toInt).reduceLeft(_ max _)
    //println("\tmax count: " + maxLabelCount)

    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    //for (l <- 0 to numLabels) {
      //val currentCase = df.filter(df("label") === l)
      val samplesToAdd = numSamples - df.count()
      //println("\t\tto add: " + samplesToAdd)
      val currentCount = df.count()
      if (0 < currentCount && currentCount < numSamples) {
        val currentSamples = df.sample(true, (numSamples - currentCount) / currentCount.toDouble).collect()
      //  println("samples created: " + currentSamples.length)
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
   // println("new count: " + samples.length)

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    val xxx = df.union(x).toDF()
    //println("joined count: " + xxx.count())
    return xxx//df.union(x).toDF()
  }

  def underSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    //println("~~~~~~~~~~~~~~~~~ Under Sample")
    var samples = Array[Row]() //FIXME - make this more parallel

      val underSampleRatio = numSamples / df.count().toDouble
      if (underSampleRatio < 1.0) {
        val currentSamples = df.sample(false, underSampleRatio).collect()
        samples = samples ++ currentSamples
        //println("new samples: " + samples.length) //FIXME - not working correctly
        val foo = spark.sparkContext.parallelize(samples)
        val x = spark.sqlContext.createDataFrame(foo, df.schema)
        return x
      }
      else {
        //println("new samples: " + df.count())
        //samples = samples ++ df.collect()
        return df
      }
  }

  def smote(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    val aggregatedCounts = df.groupBy("label").agg(count("label"))
    //println("SMOTE counts")

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0)
    //aggregatedCounts.show()
    //println("cls: " + cls + " class count: "  + currentCount)

    var smoteSamples = ArrayBuffer[Row]()
    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      //println(cls + " adding " + samplesToAdd)

      //val features = df.select("features")
      //features.printSchema()

      val currentClassZipped = df.collect().zipWithIndex

      for (s <- 1 to samplesToAdd.toInt) {
        def r = scala.util.Random.nextInt(currentClassZipped.length)

        val rand = Array(r, r, r, r, r)
        val sampled: Array[Row] = currentClassZipped.filter(x => (rand.contains(x._2))).map(x => x._1) //FIXME - issues not taking duplicates
        //FIXME - can we dump the index column?
        val values = sampled.map(x=>x(3).asInstanceOf[mutable.WrappedArray[Double]].toArray)
        val ddd = values.transpose.map(_.sum/values.length)
        val r2 = Row(0, cls, "",  ddd.toSeq)
        smoteSamples += r2
      }
    }
    else {
        // skip
    }
    samples = samples ++ smoteSamples
    val currentArray: Array[Row] = df.rdd.collect()
    samples = samples ++ currentArray
    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString().toInt, x(1).toString().toInt, x(2).toString(), x(3).asInstanceOf[mutable.WrappedArray[Double]]))

    val sqlContext = spark
    import sqlContext.implicits._
    val bar = spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minority_type")
      .withColumnRenamed("_4", "features")

    //bar2.show()
    //println("Number of added SMOTE samples: " + smoteSamples.length)
    //println("Total sampled after SMOTE : " + bar2.count())

    //println("SMOTEEEEEEEEEEEEEEEEE")
    val finalDF = underSample(spark, bar2, numSamples) //FITME - check if this is the right number

    finalDF //FIXME
  }

  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame = {
    import spark.implicits._

    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString(), x(1).toString().toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    rdd.toDF()
  }

  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect()


    val counts = getCountsByClass(spark, "label", df)
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString().toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString().toInt

//!!!println(minClassCount + " " + maxClassCount)
//!!!println("Original Train Counts: ")
//!!!counts.show()

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount / 2
    var dfs: Array[DataFrame] = Array[DataFrame]()
   // println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount)
        case _ => currentCase
      }
      dfs = dfs :+ filteredDF2
      println(filteredDF2.count())
    }

    //println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val all = dfs.reduce(_ union  _)
    println("^^^^^^ count: " + all.count())

    //println("ALL count: " + all.count())
    convertFeaturesToVector(all)

   // return all // --FIXME check if this should be removed
  }

  /*def sampleDataMinorityType(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect()

    val overSampleCount = 1000
    val underSampleCount = 200
    val smoteSampleCount = 200
    var dfs: Array[DataFrame] = Array[DataFrame]()
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
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
*/

  def minorityTypeResample(spark: SparkSession, df: DataFrame, minorityTypes: Array[String], samplingMethod: String): DataFrame = {
    //val numLabels = df.select("label").distinct().count().toInt

    //println("num labels: " + numLabels)
    //var samples = Array[Row]() //FIXME - make this more parallel

    //val types =  Array("safe", "borderline")//Array("safe", "borderline")
    //FIXME - some could be zero if split is too small
    val pickedTypes = df.filter(x => (minorityTypes contains (x(2))))

    //val d = df.select("label").distinct()
    //val presentClasses = d.select("label").rdd.map(r => r(0)).collect()

    //FIXME - avoid passing spark as parameter?
    val combinedDf = sampleData(spark, pickedTypes, samplingMethod)//sampleDataMinorityType(spark, pickedTypes, samplingMethod)

    //println("pickedType count: " + pickedTypes.count())
   // println("----------- total sampled count: " + combinedDf.count())
    //pickedTypes.show()
    //combinedDf
    //convertFeaturesToVector(combinedDf)
    combinedDf



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

    //println("test counts")
    //getCountsByClass(spark, "label", test).show()

    //println("train counts")
    //getCountsByClass(spark, "label", train).show()

    //test.show()

    val inputCols = test.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val assembledTestData = assembler.transform(test)

    val classifier = new RandomForestClassifier().setNumTrees(10).
      //setSeed(Random.nextLong()).
      setSeed(42L).
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

///!!!confusionMatrix.show()
    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum

    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))
    //classMaps.foreach(println)

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

    //  println("tp: " + tp + " fp: " + fp + " tn: " + tn + " fn: " + fn)
      val sensitivity = tp / (tp + fn).toFloat
      if (tp + fn > 0) {
        sensitiviySum += sensitivity
        sensitiviyCount += 1
      }

    }
    //println(sensitiviyCount + " " + sensitiviySum)
    println("AvAcc: " + sensitiviySum / sensitiviyCount)
  }

  import edu.vcu.sleeman.MinorityType.getMinorityTypeStatus

  def runNaiveNN(df: DataFrame, samplingMethod: String, minorityTypes: Array[Array[String]], rw: Array[String]): Unit = {
    //println("Sampled Counts")
    //val aggregatedCounts = df.groupBy("label").agg(count("label")) //FIXME
    //aggregatedCounts.show()

    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2), 42L)
    println(trainData.count() + " " + testData.count())
    //df.show()

    val minorityDF =
      if(rw(0) == "read") {
        val readData = df.sparkSession.read.
          option("inferSchema", true).
          option("header", true).
          csv(rw(1))

        val stringToArray = udf((item: String)=>item.dropRight(1).drop(1).split(",").map(x=>x.toString().toDouble))

        readData.withColumn("features", stringToArray(col("features")))
    }
    else if (rw(0) == "write") {
      val results = getMinorityTypeStatus(trainData)
        val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

        results.withColumn("features", stringify(col("features"))).
          repartition(1).
          write.format("com.databricks.spark.csv").
          option("header", "true").
          mode("overwrite").
          save(rw(1))

        results
    }
    else {
        getMinorityTypeStatus(trainData)
    }
  println("^^^^^^^^^^ minority DF")
    minorityDF.show()

    for(currentTypes<-minorityTypes) {
      print("***** ")
      for(item<-currentTypes) {
        print(item + " ")
      }
      println("***")
      println(minorityDF.count())
      //println(currentMinorityTypes.toString())
      //v/al minorityTypes = Array("safe", "borderline")
      val trainDataResampled = minorityTypeResample(minorityDF.sparkSession, minorityDF, currentTypes, samplingMethod)

      //println("trainDataResampled:" + trainDataResampled.count())
      //!!!println("Train Data Count: " + trainData.count()
      // !!!getCountsByClass(trainDataResampled.sparkSession,"label", trainDataResampled).show()

      //!!!    println("Test Data Count: " + testData.count())
      //!!!    getCountsByClass(testData.sparkSession,"label", testData).show()

      runClassifierMinorityType(trainDataResampled, testData)

    }



    //for(i<-0 to 14) {
    /*for(i<-0 to 0) {
      val currentMinorityTypes = new ArrayBuffer[String]()
      val safe = if(0 !=(i & 1)) false else { currentMinorityTypes += "safe"; true}
      val borderline = if(0 !=(i & 2)) false else {currentMinorityTypes += "borderline"; true }
      val rare = if(0 !=(i & 4)) false else { currentMinorityTypes += "rare"; true}
      val outlier = if(0 !=(i & 8)) false else { currentMinorityTypes += "outlier"; true }
     // println(safe + " " + borderline + " " + rare + " " + outlier)
      //println()
      //currentMinorityTypes.foreach(println)

    println("*** safe: [" + safe + "]\tborderline: [" + borderline + "]\trare: [" + rare + "]\toutlier: [" + outlier + "] ***")
      println(minorityDF.count())
      //println(currentMinorityTypes.toString())
      //v/al minorityTypes = Array("safe", "borderline")
      val trainDataResampled = minorityTypeResample(minorityDF.sparkSession, minorityDF, Array("safe").toArray, samplingMethod)

    //println("trainDataResampled:" + trainDataResampled.count())
//!!!println("Train Data Count: " + trainData.count()
// !!!getCountsByClass(trainDataResampled.sparkSession,"label", trainDataResampled).show()

//!!!    println("Test Data Count: " + testData.count())
//!!!    getCountsByClass(testData.sparkSession,"label", testData).show()

      runClassifierMinorityType(trainDataResampled, testData)
    }*/

  }

  def getSparkNNMinorityReport(df: DataFrame): Unit = {
    println("Minority Class Types")
    val groupedDF: DataFrame = df.select("label", "minority_type").groupBy("label", "minority_type").count()
    val listOfClasses = groupedDF.select("label").distinct().select("label").collect().map(_(0)).toList

    for(currentClass<-listOfClasses) {
      var minorityTypeMap = Map[String, Int]("safe"->0, "borderline"->0, "rare"->0, "outlier"->0)

      val currentLabel = groupedDF.filter(col("label").===(currentClass)).collect()
      for(minorityType<-currentLabel) {
        minorityTypeMap += minorityType(1).toString -> minorityType(2).toString().toInt
      }
      println("Class: " + currentClass + " safe: " + minorityTypeMap("safe") + " borderline: " + minorityTypeMap("borderline") +
        " rare: " + minorityTypeMap("rare") + "  outlier: " + minorityTypeMap("outlier"))
    }
  }
  type WI = (mutable.WrappedArray[Any], Int)


  def runSparkNN(preppedDataUpdated: DataFrame): Unit = {
    val spark = preppedDataUpdated.sparkSession
    import spark.implicits._

    val inputCols = preppedDataUpdated.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    //val assembledTestData = assembler.transform(preppedDataUpdated).select("features")

    val train_index: RDD[(Long, Row)] = preppedDataUpdated.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    train_index.take(5).foreach(println)


    val train_data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, (cls, rowMapped.reverse))
    })

    val dataRows = train_data.map(x=>(x._1, x._2._1, x._2._2.toSeq)) //FIXME - zip

    val leafSize = 5
    val knn = new KNN()
      .setTopTreeSize(dataRows.count().toInt / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setAuxCols(Array("label", "features"))

    val dfRenamed = dataRows.toDF()
      .withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")


    dfRenamed.show()
    val dfConverted = convertFeaturesToVector(dfRenamed)
    val model = knn.fit2(dfConverted).setK(6)

    val results: DataFrame = model.transform2(dfConverted)
    results.show()
    val collected: Array[Row] = results.select( "neighbors", "index", "features").collect()
    collected.take(1).foreach(println)
    //val minorityValueDF: Array[(Int, String)] = collected.map(x=>x(0)).map(x=>x.asInstanceOf[mutable.WrappedArray[Any]]).map(x=>getSparkNNMinorityResult(x))
    val minorityValueDF: Array[(Int, Int, String, Vector)] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Any]],x(1),x(2))).map(x=>getSparkNNMinorityResult(x._1, x._2.toString().toInt, x._3))

    val GGG = spark.sparkContext.parallelize(minorityValueDF).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","minority_type")
      .withColumnRenamed("_4","features")
   GGG.show()
    GGG.printSchema()
   getSparkNNMinorityReport(GGG)



    //FIXME


  }

  //def getSparkNNMinorityResult(wrappedArray: mutable.WrappedArray[Any]): (Int, String) = {
  def getSparkNNMinorityResult(x: mutable.WrappedArray[Any], index: Int, features: Any): (Int, Int, String, Vector) = {

  // val x: (mutable.WrappedArray[Any], Int) = r
    //val (x, y) = r
   // println(y)
    val wrappedArray = x

    val nearestLabels = Array[Int]()
    //println(wrappedArray.toString())
    def getLabel(neighbor: Any): Int = {
      val index = neighbor.toString().indexOf(",")
      neighbor.toString().substring(1, index).toInt
    }

    val currentLabel = getLabel(wrappedArray(0))
    var currentCount = 0
    for(i<-1 to wrappedArray.length-1) {
      nearestLabels :+ getLabel(wrappedArray(i))
      if (getLabel(wrappedArray(i)) == currentLabel) {
        currentCount += 1
      }
    }
    (index, currentLabel, getMinorityClassLabel(currentCount), features.asInstanceOf[Vector])
  }

  def main(args: Array[String]) {

    val t0 = System.nanoTime()
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    val input_file = args(0)
    val labelColumnName = args(1)

    val mode = args(2)
    val useHeader = if (args.length > 3 && args(3).equals("yes")) true else false

    val rw =
      if(args.length > 5) {
        if(args(4) == "read") Array("read", args(5).toString())
        else if(args(4)=="write") Array("write", args(5).toString())
        else Array("","")
      }
      else { Array("","") }

    println(rw)

    val df1 = spark.read.
      option("inferSchema", true).
      option("header", useHeader).
      csv(input_file)

    val df = df1.repartition(8)
    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)

    val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)

    //val samplingMethods = Array("None", "undersample", "oversample", "smote")
    val samplingMethods = Array("None")
    if(mode == "standard") {
      println("=================== Standard ====================")
      for (method <- samplingMethods) {
        println("=================== " + method + " ====================")
        runClassifier(spark, preppedDataUpdated, method)
      }
    }
    else if(mode == "naiveNN") {
      println("=================== Minority Class ====================")

      var minorityTypes = Array[Array[String]]()

      for(i<-0 to 0) {
        var currentMinorityTypes = Array[String]()
        if (0 != (i & 1)) false else {
          currentMinorityTypes = currentMinorityTypes :+ "safe"
        }
        if (0 != (i & 2)) false else {
          currentMinorityTypes = currentMinorityTypes :+ "borderline"
        }
        if (0 != (i & 4)) false else {
          currentMinorityTypes = currentMinorityTypes :+ "rare"
        }
        if (0 != (i & 8)) false else {
          currentMinorityTypes = currentMinorityTypes :+ "outlier"
        }
        minorityTypes = minorityTypes :+ currentMinorityTypes
      }

      for (method <- samplingMethods) {
        println("=================== " + method + " ====================")
          runNaiveNN(preppedDataUpdated, method, minorityTypes, rw)
      }
    }
    else if(mode == "sparkNN") {
      runSparkNN(preppedDataUpdated)
    }
    else {
      println("ERROR: running mode " + mode + " is not valid [standard, naiveNN, sparkNN")
    }

    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1e9 + "s")

  }
}
