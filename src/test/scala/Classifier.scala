package edu.vcu.sleeman

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame
import java.io.PrintWriter
import java.io.File
import org.apache.log4j._
import scala.collection.mutable
import org.apache.spark.ml.classification._
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.knn.KNN.VectorWithNorm
import org.apache.spark.ml.clustering.KMeans


//FIXME - turn classes back to Ints instead of Doubles
object Classifier {

  val kValue = 20 //FIXME
  var results = ""

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

  def maxValue(a: Double, b:Double): Double ={
    if(a >= b) return a
    else return b
  }

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame): String ={
    import distinctClasses.sparkSession.implicits._
    //FIXME - don't calculate twice
    val maxLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString().toDouble.toInt).max
    val minLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString().toDouble.toInt).min
    val numberOfClasses = distinctClasses.count()
    val classCount = confusionMatrix.columns.length - 1
    val testLabels = distinctClasses.map(_.getAs[Int]("label")).map(x => x.toInt).collect().sorted

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum
    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))

    confusionMatrix.show(60,false)

    var AvAvg = 0.0
    var MAvG = 1.0
    var RecM = 0.0
    var PrecM = 0.0
    var Precu = 0.0
    var Recu = 0.0
    var FbM = 0.0
    var Fbu = 0.0
    var AvFb = 0.0
    var CBA = 0.0

    var tSum = 0.0
    var pSum = 0.0
    var tpSum = 0.0
    val beta = 0.5 // User specified

    //FIXME - could be made parallel w/udf
    //for (clsIndex <- 1 to maxLabel) {
    for (clsIndex <- minLabel to maxLabel - minLabel) {
      print(clsIndex + "\n")
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0
      val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x(0).toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn: Double = colSum - tp
      val fp: Double = rowValueSum - tp
      val tn: Double = totalCount - tp - fp - fn

      val recall = tp / (tp + fn)
      val precision = tp / (tp + fp)

      println("class acc: " + ((tp + tn) / (tp + tn + fp + fn)))
      println(tp + " " + tn + " " + fp + " " + fn)

      AvAvg += ((tp + tn) / (tp + tn + fp + fn))
      MAvG *= recall
      RecM += { if(recall.isNaN()) 0.0 else recall }
      PrecM += precision
      def getAvFb(): Double= {
        val result = (((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall))
        if(result.isNaN()) {
          0.0
        }
        else result
      }
      AvFb += getAvFb()

      //FIXME - what to do if col/row sum are zero?
      var rowColMaxValue = maxValue(colSum, rowValueSum)
      if(rowColMaxValue > 0) {
        CBA += tp / rowColMaxValue

        println("CBA value: " + tp / rowColMaxValue)//maxValue(colSum, rowValueSum))
      }
      else {
        println("CBA value NaN")
      }

      //CBA += (tp / maxValue(colSum, rowValueSum))



      // for Recu and Precu
      tpSum += tp
      tSum += (tp + fn)
      pSum += (tp + fp)
    }

    println("class count"  + numberOfClasses.toDouble)
    AvAvg /= classCount//numberOfClasses.toDouble
    MAvG = {  val result = Math.pow((MAvG), (1/numberOfClasses.toDouble)); if(result.isNaN()) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= classCount//numberOfClasses
    PrecM /= classCount//numberOfClasses
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN()) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN()) 0.0 else result }
    AvFb /= classCount//numberOfClasses.toDouble
    CBA /= classCount//numberOfClasses.toDouble

    AvAvg  + "," + MAvG + "," + RecM +"," + PrecM + "," + Recu + "," + Precu + "," + FbM + "," + Fbu + "," + AvFb + "," + CBA
  }

  def runClassifier(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterResults: Map[Int,DataFrame], enableDataScaling: Boolean): String = {
    import spark.implicits._
    //val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2),42L)
    //FIXME - add cross validation
    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, "", rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")

      // Compute summary statistics and generate MinMaxScalerModel
      val scalerModel = scaler.fit(converted)

      // rescale each feature to range [min, max].
      //val scaledData = scalerModel.transform(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData
    } else {
      converted
    }

    val Array(trainData, testData) = scaledData.randomSplit(Array(0.8, 0.2),42L)
    trainData.printSchema()

    val trainDataSampled = sampleData(spark, trainData, samplingMethod, clusterResults)
    //getCountsByClass(spark, "label", trainDataSampled)

    val maxLabel: Int = testData.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).max

    println("test counts")
    getCountsByClass(spark, "label", testData).show()

    println("train counts")
    getCountsByClass(spark, "label", trainDataSampled).show()


    println("** train data **")
    trainDataSampled.show()

    val classifier = new RandomForestClassifier().setNumTrees(10).
      //setSeed(Random.nextLong()).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

        val model = classifier.fit(trainDataSampled)
     val predictions2 = model.transform(testData)

     val confusionMatrix = predictions2.
       groupBy("label").
       pivot("prediction", (0 to maxLabel)).
       count().
       na.fill(0.0).
       orderBy("label")

    confusionMatrix.show()

    samplingMethod + ",," + calculateClassifierResults(testData.select("label").distinct(), confusionMatrix) // FIXME - why to ,'s?
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
    val cls = current._2._1
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

  def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distsance = 0.0
    for(index<-0 to x.length-1) {
      distsance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distsance
  }

  def getSmoteSample(data: Array[Array[Double]]): Unit = {
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    val results = data.map(x=>Vectors.dense(x.map(_.toDouble).toArray))
    //println("DISTANCE: " +  Math.sqrt(fastSquaredDistance(results(0), Vectors.norm(results(0), 2), results(1), Vectors.norm(results(1), 2))))
    val foo = new VectorWithNorm(results(0))
    val foo2 = new VectorWithNorm(results(1))
    //println(fastSquaredDistance(foo.vector, foo.norm, foo2.vector, foo2.norm))
    //println("***")
    //println(foo.fastSquaredDistance(foo2))
    //println(getSingleDistance(data(0), data(1)))

    //fastSquaredDistance
    //data.map(x=>data.map(y=>getSingleDistance(x, y)))
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
    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
      val samplesToAdd = numSamples - df.count()
      val currentCount = df.count()
      if (0 < currentCount && currentCount < numSamples) {
        val currentSamples = df.sample(true, (numSamples - currentCount) / currentCount.toDouble).collect()
        samples = samples ++ currentSamples
      }

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    df.union(x).toDF()
  }

  def underSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel

      val underSampleRatio = numSamples / df.count().toDouble
      if (underSampleRatio < 1.0) {
        val currentSamples = df.sample(false, underSampleRatio, seed = 42L).collect()
        samples = samples ++ currentSamples
        val foo = spark.sparkContext.parallelize(samples)
        val x = spark.sqlContext.createDataFrame(foo, df.schema)
        return x
      }
      else {
         return df
      }
  }

  def getDistance(a: Array[Double], b: Array[Double]): Double = {
    val zipped = a.zip(b)
    val result = zipped.map({ case (x, y) => (x - y) * (x - y) })
    Math.sqrt(result.sum)
  }

  def getAverageDistance(x: Array[Double], array: Array[Array[Double]]): Double ={
    array.map(y=>getDistance(x,y)).sum/(array.length-1)
  }

  /*
  def getAverageDistances(data:DataFrame): Double ={
    import data.sparkSession.implicits._
    val collectedData = data.select("features").collect().map(x=>x(0).asInstanceOf[mutable.WrappedArray[Double]].toArray)


    val results: Dataset[Double] = data.select("features").map(x=>getAverageDistance(x(0).asInstanceOf[mutable.WrappedArray[Double]].toArray, collectedData))
    results.show()
    val averageDistance = results.agg(sum("value")).collect()(0)(0).asInstanceOf[Double]/collectedData.length


   // collectedData.show()
    println("SMOTE: " + averageDistance)
    //averageDistance.show()
    //collectedData(0).foreach(println)

    0.0 //FIXME -??
  }*/

  def smote(spark: SparkSession, df: DataFrame, numSamples: Int, predictions: DataFrame, sparkMeans: Boolean=false): DataFrame = {
    val aggregatedCounts = df.groupBy("label").agg(count("label"))

    val randomInts = new scala.util.Random(42L)

    println("************* @ smote")

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0)

    println(cls + " "+ currentCount + " " + numSamples)

    var smoteSamples = ArrayBuffer[Row]()
    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      val currentClassZipped = df.collect().zipWithIndex

        if(sparkMeans) {
          /*//FIXME - pick number of K
          println("** DF ** ")
          df.show()
          val kValue = 5
          val kmeans = new KMeans().setK(kValue).setSeed(1L)
          val model2 = kmeans.fit(df)

          // Make predictions
          val predictions = model2.transform(df).select("prediction", "index", "label", "minorityType", "features").cache()
          */
          predictions.show()
          predictions.printSchema()
          val predictionsCollected = predictions.collect().map(x=>(x(0).toString().toInt, x(1).toString().toInt, x(2).toString().toInt, x(3).toString(), x(4).asInstanceOf[DenseVector])).toSeq
          val clusteredData = predictionsCollected.groupBy {_._1}

          for (s <- 1 to samplesToAdd.toInt) {

            def clusterIndex = randomInts.nextInt(kValue)//scala.util.Random.nextInt(kValue)
                        def r = randomInts.nextInt(clusteredData(clusterIndex).length) //scala.util.Random.nextInt(clusteredData(clusterIndex).length)
            def getSample = clusteredData(clusterIndex)(randomInts.nextInt(5)) //clusteredData(clusterIndex)(scala.util.Random.nextInt(5))
            //val rand = Array(r, r, r, r, r)
            val sampled = Array.fill(kValue)(getSample)//Array(getSample, getSample, getSample, getSample, getSample)

            //val sampled: Array[Row] = currentClassZipped.filter(x => (rand.contains(x._2))).map(x => x._1) //FIXME - issues not taking duplicates - might be fixed?

            //FIXME - can we dump the index column?
            val values: Array[Array[Double]] = sampled.map(x=>x._5.asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

            val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
            val r2 = Row(0, cls, "",  Vectors.dense(ddd.map(_.toDouble)))

            //FIXME - convert this to DenseVector
            smoteSamples += r2
          }
        }
        else {
          for (s <- 1 to samplesToAdd.toInt) {
          def r = randomInts.nextInt(currentClassZipped.length) //scala.util.Random.nextInt(currentClassZipped.length)

            val rand = Array(r, r, r, r, r)
            val sampled: Array[Row] = currentClassZipped.filter(x => (rand.contains(x._2))).map(x => x._1) //FIXME - issues not taking duplicates
            //FIXME - can we dump the index column?
            val values: Array[Array[Double]] = sampled.map(x=>x(3).asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

            val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
            val r2 = Row(0, cls, "",  Vectors.dense(ddd.map(_.toDouble)))

            //FIXME - convert this to DenseVector
            smoteSamples += r2
          }
        }
    }
    else {
        // skip
    }
   df.show()
    df.printSchema()

    samples = samples ++ smoteSamples
    val currentArray = df.rdd.map(x=>Row(x(0), x(1), x(2), x(3).asInstanceOf[DenseVector])).collect()
    println(currentArray.take(1)(0))

    samples = samples ++ currentArray

    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString().toInt, x(1).toString().toInt, x(2).toString(), x(3).asInstanceOf[DenseVector]))//asInstanceOf[mutable.WrappedArray[Double]]))

    import df.sparkSession.implicits._
    val bar = spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
    bar2.show()

    val finalDF = underSample(spark, bar2, numSamples) //FITME - check if this is the right number

    finalDF.show()

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

  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {
    println("~~~~~ sampleData ~~~~~")
    df.printSchema()
    getCountsByClass(spark, "label", df).show()
    val d = df.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString().toInt)

    val counts = getCountsByClass(spark, "label", df)
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString().toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString().toInt

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount / 2
    var dfs: Array[DataFrame] = Array[DataFrame]()

    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount, clusterResults(l), false)
        case "smotePlus" => smote(spark, currentCase, smoteSampleCount, clusterResults(l),true)
        case _ => currentCase
      }
      dfs = dfs :+ filteredDF2
    }

    val all = dfs.reduce(_ union  _)
    //convertFeaturesToVector(all)
    all
  }

  def minorityTypeResample(spark: SparkSession, df: DataFrame, minorityTypes: Array[String], samplingMethod: String, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {
    //FIXME - some could be zero if split is too small
    val pickedTypes = df.filter(x => (minorityTypes contains (x(2))))

    //FIXME - avoid passing spark as parameter?
    val combinedDf = sampleData(spark, pickedTypes, samplingMethod, clusterResults, cutoff)
    combinedDf
  }

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }

  def runClassifierMinorityType(train: DataFrame, test: DataFrame): String = {
    val spark = train.sparkSession
    //FIXME - don't collect twice
    val maxLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).max
    val minLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString().toDouble.toInt).min
    val inputCols = test.columns.filter(_ != "label")

    val classifier = new RandomForestClassifier().setNumTrees(10).
      //setSeed(Random.nextLong()).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

    train.show()
    train.printSchema()

    val model = classifier.fit(train)
    val predictions = model.transform(test)
    //val testLabels = test.select("label").distinct().map(_.getAs[Double]("label")).map(x => x.toInt).collect().sorted

    val confusionMatrix = predictions.
      groupBy("label").
      pivot("prediction", (minLabel to maxLabel)).
      //pivot("prediction", (0 to maxLabel)).
      count().
      na.fill(0.0).
      orderBy("label")

    calculateClassifierResults(test.select("label").distinct(), confusionMatrix)
  }

  import edu.vcu.sleeman.MinorityType.{getMinorityTypeStatus, getMinorityTypeStatus2}

  def runNaiveNN(df: DataFrame, samplingMethod: String, minorityTypes: Array[Array[String]], clusterResults: Map[Int,DataFrame], enableDataScaling: Boolean, rw: Array[String]): String = {
    df.show()

    import df.sparkSession.implicits._
    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })


    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      //.withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
    } else { converted }

    val Array(trainData, testData) = scaledData.randomSplit(Array(0.8, 0.2),42L)

    trainData.show()
    val minorityDF = getMinorityTypeStatus2(trainData)

    getSparkNNMinorityReport(minorityDF)
    var currentResults = ""
    for(currentTypes<-minorityTypes) {
      var currentTypesString = "["
      for(item<-currentTypes) {
        currentTypesString += item + " "
      }
      currentTypesString = currentTypesString.substring(0, currentTypesString.length()-1)
      currentTypesString += "]"
      val trainDataResampled = minorityTypeResample(minorityDF.sparkSession, minorityDF, currentTypes, samplingMethod, clusterResults)

      currentResults += samplingMethod + "," + currentTypesString + ","
      currentResults += runClassifierMinorityType(convertFeaturesToVector(trainDataResampled), testData) + "\n"
    }
    currentResults
  }

  def getSparkNNMinorityReport(df: DataFrame): Unit = {
    println("Minority Class Types")
    val groupedDF: DataFrame = df.select("label", "minorityType").groupBy("label", "minorityType").count()
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
  def runSparkNN(trainData: DataFrame, testData: DataFrame, minorityDF: DataFrame, samplingMethod: String, minorityTypes: Array[Array[String]], clusterResults: Map[Int,DataFrame], enableDataScaling: Boolean): String = {
    println("^^^ train ^^^^")
    getCountsByClass(trainData.sparkSession, "label", trainData).show()
    println("^^^ test ^^^^")
    getCountsByClass(testData.sparkSession, "label", testData).show()


    println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    trainData.printSchema()

    var currentResults = ""
    //FIXME
    for(currentTypes<-minorityTypes) {
      var currentTypesString = "["
      for (item <- currentTypes) {
        currentTypesString += item + " "
      }
      currentTypesString = currentTypesString.substring(0, currentTypesString.length()-1)

      currentTypesString += "]"
      val trainDataResampled = minorityTypeResample(minorityDF.sparkSession, convertFeaturesToVector(minorityDF), currentTypes, samplingMethod, clusterResults, 0.0)
      trainDataResampled.printSchema()
      currentResults += samplingMethod + "," + currentTypesString + ","
      //currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
      currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
    }
    currentResults
  }

  def getSparkNNMinorityResult(x: mutable.WrappedArray[Any], index: Int, features: Any): (Int, Int, String, mutable.WrappedArray[Double]) = {
    val wrappedArray = x

    val nearestLabels = Array[Int]()
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
    val currentArray = features.toString().substring(1, features.toString().length-1).split(",").map(x=>x.toDouble)
    (index, currentLabel, getMinorityClassLabel(currentCount), currentArray)//features.toString().asInstanceOf[mutable.WrappedArray[Double]])
  }

  def main(args: Array[String]) {

    val t0 = System.nanoTime()
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    print(args.length)
    for(x<- args) {
      print(x + '\n')
    }
    if (args.length < 5) {
      print("Usage: imbalanced-spark [input file path] [label column] [useHeader] [save path] [r/w mode] [r/w path]")
      return
    }

    val input_file = args(0)
    val labelColumnName = args(1)

    val mode = args(2)
    val useHeader = if (args.length > 3 && args(3).equals("yes")) true else false

    val savePath = args(4)

    val rw =
      if(args.length > 6) {
        if(args(5) == "read") Array("read", args(6).toString())
        else if(args(5)=="write") Array("write", args(6).toString())
        else Array("","")
      }
      else { Array("","") }

    val df1 = spark.read.
      option("inferSchema", true).
      option("header", useHeader).
      csv(input_file)

    val df = df1.repartition(8)
    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)

    val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)
    var minorityTypes = Array[Array[String]]()

    //for(i<-0 to 14) {
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

    /*************************************************************/
    import df.sparkSession.implicits._

    val enableDataScaling = true

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      //.withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
    } else { converted }


    //scaledData.show(50)
    //scaledData.filter(scaledData("index") < 25).show(50)
    //val foo = scaledData.filter(scaledData("index") < 100000)

    val numSplits = 5
    val counts = scaledData.count()
    var cuts = Array[Int]()
    println("count: " + counts)
    cuts :+= 0
    for(i <- 1 to (numSplits - 1)) {
      cuts :+= ((counts / numSplits) * i).toInt
      //cuts :+= 2000

    }
    cuts :+= counts.toInt

    //cuts :+=0
    //cuts :+=2000
    //cuts :+= counts.toInt

    cuts.foreach(println)


    for(cutIndex<-0 to cuts.length-2) {
      println(cuts(cutIndex) + " " + (cuts(cutIndex+1)))

      val testData = scaledData.filter(scaledData("index") < cuts(cutIndex+1) && scaledData("index") >= cuts(cutIndex))
      val trainData = scaledData.filter(scaledData("index") >= cuts(cutIndex+1) || scaledData("index") < cuts(cutIndex))

      println("train: " + trainData.count() )
      println("test: " + testData.count() )

      //val Array(trainData, testData) = scaledData.randomSplit(Array(0.8, 0.2),42L)
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      trainData.printSchema()

    /*************************************************************/

      val leafSize = 5
      val knn = new KNN()
        .setTopTreeSize(scaledData.count().toInt / 10)
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
        .setSeed(42L)
        .setAuxCols(Array("label", "features"))
      val model = knn.fit(scaledData).setK(6)//.setDistanceCol("distances")
      val results2: DataFrame = model.transform(scaledData)

      val collected: Array[Row] = results2.select( "neighbors", "index", "features").collect()
      results2.show()
      val minorityValueDF: Array[(Int, Int, String, mutable.WrappedArray[Double])] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Any]],x(1),x(2))).map(x=>getSparkNNMinorityResult(x._1, x._2.toString().toInt, x._3))

      val minorityDF = scaledData.sparkSession.sparkContext.parallelize(minorityValueDF).toDF()
        .withColumnRenamed("_1","index")
        .withColumnRenamed("_2","label")
        .withColumnRenamed("_3","minorityType")
        .withColumnRenamed("_4","features")//.sort("index")
      getSparkNNMinorityReport(minorityDF)

    /*************************************************************/


    //val samplingMethods = Array("none", "undersample", "oversample", "smote", "smotePlus")//)//, "smote", "smotePlus")//, "smote", "smotePlus")
    val samplingMethods = Array("smote")

    val d = trainData.select("label").distinct()
    val presentClasses = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString().toInt)


      def getClassClusters(l: Int, df: DataFrame): (Int, DataFrame) = {
        spark.sqlContext.emptyDataFrame
        val currentCase = df.filter(df("label") === l).toDF()
        //FIXME - pick number of K
        //println("** At kmeans ** ")
        //currentCase.show()
        //val kValue = 5
        val kmeans = new KMeans().setK(kValue).setSeed(1L)
        val convertedDF = convertFeaturesToVector(minorityDF)
        val model2 = kmeans.fit(convertedDF)
        // Make predictions
        val predictions = model2.transform(convertedDF).select("prediction", "index", "label", "minorityType", "features").cache()
        //predictions.show()
        (l, predictions)
      }


      val clusterResults: Map[Int, DataFrame] = presentClasses.map(x=>getClassClusters(x, minorityDF)).toMap
      /*
      val predictions = if(samplingMethods.contains("smotePlus")) {
        spark.sqlContext.emptyDataFrame
        //FIXME - pick number of K
          println("** At kmeans ** ")
        minorityDF.show()
          val kValue = 5
          val kmeans = new KMeans().setK(kValue).setSeed(1L)
          val model2 = kmeans.fit(minorityDF)

          // Make predictions
          val predictions = model2.transform(minorityDF).select("prediction", "index", "label", "minorityType", "features").cache()
          predictions.show()
          predictions
      }
    else {
      spark.sqlContext.emptyDataFrame
    }*/

      //Array("none", "undersample", "oversample", "smote", "smotePlus")//, "undersample", "oversample", "smote")
      if(mode == "standard") {
        val writer = new PrintWriter(new File("/home/ford/repos/imbalanced-spark/standard.txt"))
        //writer.write("sampling,minorityTypes,AvAcc\n")
        writer.write("sampling,minorityTypes,AvAvg,MAvG,RecM,PrecM,Recu,Precu,FbM,Fbu,AvFb,CBA\n")
        println("=================== Standard ====================")
        for (method <- samplingMethods) {
          println("=================== " + method + " ====================")
          writer.write(runClassifier(spark, preppedDataUpdated, method, clusterResults,true) + "\n")
        }
        writer.close()
      }
      else if(mode == "naiveNN") {
        val writer = new PrintWriter(new File("/home/ford/repos/imbalanced-spark/naiveNN.txt"))
        writer.write("sampling,minorityTypes,AvAvg,MAvG,RecM,PrecM,Recu,Precu,FbM,Fbu,AvFb,CBA\n")
        println("=================== Minority Class ====================")
        for (method <- samplingMethods) {
          println("=================== " + method + " ====================")
          writer.write(runNaiveNN(preppedDataUpdated, method, minorityTypes, clusterResults, true, rw))
        }
        writer.close()
      }
      else if(mode == "sparkNN") {
        val writer = new PrintWriter(new File(savePath + cutIndex.toString))///home/ford/repos/imbalanced-spark/sparkNN.txt"))
        writer.write("sampling,minorityTypes,AvAvg,MAvG,RecM,PrecM,Recu,Precu,FbM,Fbu,AvFb,CBA\n")
        for (method <- samplingMethods) {
          //writer.write(runSparkNN(preppedDataUpdated, method, minorityTypes, true))
          writer.write(runSparkNN(trainData, testData, minorityDF, method, minorityTypes, clusterResults, true))
        }
        writer.close()
      }
      else {
        println("ERROR: running mode " + mode + " is not valid [standard, naiveNN, sparkNN")
      }


    }









    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1e9 + "s")
  }
}
