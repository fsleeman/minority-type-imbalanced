package edu.vcu.sleeman

import org.apache.spark.ml.Pipeline
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame
import java.io.File

import org.apache.log4j._

import scala.collection.mutable
import org.apache.spark.ml.classification._
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD

import scala.collection.parallel.immutable.ParSeq
import scala.util.{Failure, Random, Success, Try}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel


//FIXME - turn classes back to Ints instead of Doubles
object Classifier {

  var resultIndex = 0
  var kValue:Int = 0

  val clusterKValues: Array[Int] = Array(2, 5, 10, 25, 50)
  val cutoffs: Array[Double] = Array(0.0)
  var results = ""

  var resultArray: Array[Array[String]] = Array()

  type NearestClassResult = (Int, Array[Int]) // class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) // index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)


  def getIndexedDF(spark: SparkSession, columns: Array[String], trainData: DataFrame): DataFrame = {
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
    if(a >= b) { a }
    else { b }
  }

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame): Array[String]={
    import distinctClasses.sparkSession.implicits._
    val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt)

    val maxLabel: Int = classLabels.max
    val minLabel: Int = classLabels.min
    val numberOfClasses = classLabels.length
    val classCount = confusionMatrix.columns.length - 1
    val testLabels = distinctClasses.map(_.getAs[Int]("label")).map(x => x.toInt).collect().sorted

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum
    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))

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
    for (clsIndex <- minLabel to maxLabel - minLabel) {
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0
      val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn: Double = colSum - tp
      val fp: Double = rowValueSum - tp
      val tn: Double = totalCount - tp - fp - fn

      val recall = tp / (tp + fn)
      val precision = tp / (tp + fp)

      AvAvg += ((tp + tn) / (tp + tn + fp + fn))
      MAvG *= recall
      RecM += { if(recall.isNaN) 0.0 else recall }
      PrecM += precision

      val getAvFb: Double= {
        val result = ((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall)
        if(result.isNaN) {
          0.0
        }
        else result
      }
      AvFb += getAvFb

      //FIXME - what to do if col/row sum are zero?
      val rowColMaxValue = maxValue(colSum, rowValueSum)
      if(rowColMaxValue > 0) {
        CBA += tp / rowColMaxValue
      }
      else {
        //println("CBA value NaN")
      }

      // for Recu and Precu
      tpSum += tp
      tSum += (tp + fn)
      pSum += (tp + fp)
    }

    AvAvg /= classCount
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= classCount
    PrecM /= classCount
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN) 0.0 else result }
    AvFb /= classCount
    CBA /= classCount

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount: Int): Int = {

    if (kCount / kValue.toFloat >= 0.8) { 0 }
    else if ( kCount / kValue.toFloat >= 0.4) { 1 }
    else if ( kCount / kValue.toFloat >= 0.2) { 2 }
    else { 3 }
  }

  def setMinorityStatus(cls: Int, sample: (Int, Array[Int])): (Int, Int) = {
    // positive case
    if (cls == sample._1) {
      val matchingCount = sample._2.count(x => x == cls)
      (1, getMinorityClassLabel(matchingCount))
    }
    // negative case
    else {
      val matchingCount = sample._2.count(x => x != cls)
      (0, getMinorityClassLabel(matchingCount))
    }
  }

  def getMinorityClassResults(cls: Int, name: String, data: Array[NearestClassResult]) {
    val minorityStatus = data.map(x => setMinorityStatus(cls, x))

    val positiveSamples = minorityStatus.filter(x => x._1 == 1).map(x => x._2)
    val negativeSamples = minorityStatus.filter(x => x._1 != 1).map(x => x._2)

    val positiveSafeCount = positiveSamples.count(x => x == "safe")
    val positiveBorderlineCount = positiveSamples.count(x => x == "borderline")
    val positiveRareCount = positiveSamples.count(x => x == "rare")
    val positiveOutlierCount = positiveSamples.count(x => x == "outlier")
    val positiveResults = "Positive: " + positiveSamples.length + "\nsafe: " + positiveSafeCount + " borderline: " + positiveBorderlineCount + " rare: " + positiveRareCount + " outlier: " + positiveOutlierCount

    val negativeSafeCount = negativeSamples.count(x => x == "safe")
    val negativeBorderlineCount = negativeSamples.count(x => x == "borderline")
    val negativeRareCount = negativeSamples.count(x => x == "rare")
    val negativeOutlierCount = negativeSamples.count(x => x == "outlier")
    val negativeResults = "Negative: " + negativeSamples.length + "\nsafe: " + negativeSafeCount + " borderline: " + negativeBorderlineCount + " rare: " + negativeRareCount + " outlier: " + negativeOutlierCount

    println("\nClass " + name + "\n" + positiveResults + "\n" + negativeResults + "\n")
  }

  def getDistances2(current: Element, train: Array[Element]): NearestClassResultIndex = {
    val result = train.map(x => getDistanceValue(x, current)).sortBy(x => x._1).take(5)
    (current._1, current._2._1, result.map(x => x._2))
  }

  def getDistanceValue(train: Element, test: Element): DistanceResult = {
    if (train._1 == test._1) {
      (Float.MaxValue, train._2._1)
    }
    else {
      val zipped = test._2._2.zip(train._2._2)
      val result = zipped.map({ case (x, y) => (x - y) * (x - y) })
      (result.sum, train._2._1) //removed sqrt
    }
  }

  def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  def mapRow(currentRow: Array[Any]): (Int, Array[Float]) = {
    val reverseRow = currentRow.reverse
    val cls = reverseRow.head.toString.toFloat.toInt
    val features = reverseRow.tail.map(_.toString.toFloat)
    (cls, features)
  }

  def convertIndexedToName(cls: Int, indexedLabelNames: DataFrame): String = {
    val rows: Array[Row] = indexedLabelNames.collect
    val res = rows.filter(x => x(0) == cls)
    res(0)(1).toString
  }

  def getIndexedLabelNames(df: DataFrame): DataFrame = {
    val converter = new IndexToString()
      .setInputCol("label")
      .setOutputCol("originalCategory")
    val converted = converter.transform(df)
    converted.select("label", "originalCategory").distinct()
  }

  // assume there is only one class present
  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]()
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
      x
    }
    else {
      df
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

  def smoteSample(randomInts: Random, currentClassZipped: Array[(Row, Int)], cls: Int): Row = {
    def r = randomInts.nextInt(currentClassZipped.length)

    val rand = Array(r, r, r, r, r)
    val sampled: Array[Row] = currentClassZipped.filter(x => rand.contains(x._2)).map(x => x._1)
    val values: Array[Array[Double]] = sampled.map(x=>x(3).asInstanceOf[DenseVector].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)

    val r2 = Row(0, cls, -1,  Vectors.dense(ddd.map(_.toDouble)))
    r2
  }

  def smote(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    println("Previous DF Schema")
    df.printSchema()
    val aggregatedCounts = df.groupBy("label").agg(count("label"))

    val randomInts: Random = new scala.util.Random(42L)
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0).toString().toInt //FIXME
    println(cls)

    println("numSamples: " + numSamples)
    val finaDF = if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val currentClassZipped = df.collect().zipWithIndex

      val mappedResults = spark.sparkContext.parallelize(1 to samplesToAdd.toInt).map(x => smoteSample(randomInts, currentClassZipped, cls))
      val mappedDF = spark.sqlContext.createDataFrame(mappedResults, df.schema)

      val joinedDF = df.union(mappedDF)
      println("   ~~~ NEW SCHEMA ~~~")
      joinedDF.printSchema()
      joinedDF
    }
    else {
      df
    }
    finaDF
  }

    def smoteOrig(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {

    println("Previous DF Schema")
    df.printSchema()
    val aggregatedCounts = df.groupBy("label").agg(count("label"))

    val randomInts: Random = new scala.util.Random(42L)

    var samples = ArrayBuffer[Row]()
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0).toString().toInt
    println(cls)

    var smoteSamples = ArrayBuffer[Row]()
    println("numSamples: " + numSamples)

    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val currentClassZipped = df.collect().zipWithIndex

      for (s <- 1 to samplesToAdd.toInt) {
        def r = randomInts.nextInt(currentClassZipped.length) //scala.util.Random.nextInt(currentClassZipped.length)

        val rand = Array(r, r, r, r, r)
        val sampled: Array[Row] = currentClassZipped.filter(x => rand.contains(x._2)).map(x => x._1)
        val values: Array[Array[Double]] = sampled.map(x=>x(3).asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

        val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
        val r2 = Row(0, cls, "",  Vectors.dense(ddd.map(_.toDouble)))

        smoteSamples += r2
      }
    }

    else {
      // we already have enough samples, skip
    }

    samples = samples ++ smoteSamples
    val currentArray = df.rdd.map(x=>Row(x(0), x(1), x(2), x(3).asInstanceOf[DenseVector])).collect()
    samples = samples ++ currentArray

    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString, x(3).asInstanceOf[DenseVector]))//asInstanceOf[mutable.WrappedArray[Double]]))

    import df.sparkSession.implicits._
    val bar = spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
    bar2
  }

  def YYY(cls: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random): Row = {
    val clusterIndex = clusterIds(randomInts.nextInt(clusterIds.length))
    val sampled: Array[(Int, Int, Int, Int, DenseVector)] = Array.fill(clusterCount)(clusteredData(clusterIndex)(randomInts.nextInt(clusteredData(clusterIndex).length)))

    val values: Array[Array[Double]] = sampled.map(x=>x._5.asInstanceOf[DenseVector].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
    val r2 = Row(0, cls, 0,  Vectors.dense(ddd.map(_.toDouble)))
    r2
  }

  def smotePlus(spark: SparkSession, df: DataFrame, numSamples: Int, predictions: DataFrame, l: Int, clusterCount: Int): DataFrame = {
    val randomInts: Random = new scala.util.Random(42L)

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount: Long = df.count()
    println("***^ INSIDE current count: "  + l + " " + currentCount)
    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount

      println("At spark Means " + samplesToAdd)

      val predictionsCollected = predictions.collect().map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).toString.toInt, x(4).asInstanceOf[DenseVector])).toSeq
      val clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]] = predictionsCollected.groupBy {_._1}
      val clusterIds = clusteredData.keySet.toSeq  //.map(x=>x._1).toSeq

      println("samplesToAdd: "  + samplesToAdd)

      val t01 = System.nanoTime()
      val XXX: ParSeq[Row] = (1 to samplesToAdd.toInt).par.map { _ => YYY(l: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random) }// .reduce(_ ++ _)
      samples = samples ++ XXX
      val t11 = System.nanoTime()
      println("--------- LOOP TIME: " + (t11 - t01) / 1e9 + "s")

    }
    else {
      // we already have enough samples, skip
    }

    val tX = System.nanoTime()
    val currentArray = df.rdd.map(x=>Row(x(1), x(2), x(3), x(4).asInstanceOf[DenseVector])).collect()
    samples = samples ++ currentArray
    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).asInstanceOf[DenseVector]))

    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
    val tY = System.nanoTime()
    println("--------- Combine Time: " + (tY - tX) / 1e9 + "s")
    bar2
  }

  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame = {
    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString.toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    spark.createDataFrame(rdd)
  }

  def sampleDataParallel(spark: SparkSession, df: DataFrame, presentClass: Int, samplingMethod: String, underSampleCount: Int, overSampleCount: Int, smoteSampleCount: Int): DataFrame = {
      val l = presentClass
      println("----------> Class: " + presentClass)
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount)
        case _ => currentCase
      }

    filteredDF2
  }

  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

    val counts = getCountsByClass(spark, "label", df)
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString.toInt

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount / 2

    val myDFs: Array[(Int, DataFrame)] = presentClasses.map(x=>(x, df.filter(df("label") === x).toDF()))
    val classDF: Array[DataFrame] = presentClasses.map(x => sampleDataParallel(spark, myDFs.filter(y=>y._1 == x)(0)._2, x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))

    println("Final count ")
    println(classDF.length)
    val r = classDF.reduce(_ union _)
    r
  }

  def sampleDataSmotePlus(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterKValue: Int, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {
    val d = df.select("label").distinct()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    d.show()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt) //Array(1,2,3,4,5,6,7)    //

    val counts = getCountsByClass(spark, "label", df)
    println("***^ COUNTS OUTSIDE")
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val smoteSampleCount = maxClassCount / 2
    var dfs: Array[DataFrame] = Array[DataFrame]()

    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF() ///FIXME - this may already been calculated
      val filteredDF2 = samplingMethod match {
        case "smotePlus" => smotePlus(spark, currentCase, smoteSampleCount, clusterResults(l), l ,clusterKValue)
        case _ => currentCase
      }
      dfs = dfs :+ filteredDF2
    }

    val all = dfs.reduce(_ union  _)
    all
  }

  def minorityTypeResample(spark: SparkSession, df: DataFrame, minorityTypes: Array[Int], samplingMethod: String): DataFrame = {
    val pickedTypes = df.filter(x => minorityTypes contains x(2))
    sampleData(spark, pickedTypes, samplingMethod)
  }

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }

  def runClassifierMinorityType(train: DataFrame, test: DataFrame): Array[String] ={//String = {
    val spark = train.sparkSession
    //FIXME - don't collect twice
    val maxLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble.toInt).max
    val minLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble.toInt).min
    val inputCols = test.columns.filter(_ != "label")

    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

    val model = classifier.fit(train)
    val predictions = model.transform(test)

    val confusionMatrix: Dataset[Row] = predictions.
      groupBy("label").
      pivot("prediction", minLabel to maxLabel).
      count().
      na.fill(0.0).
      orderBy("label")

    calculateClassifierResults(test.select("label").distinct(), confusionMatrix)
  }

  def runSparkNN(trainData: DataFrame, testData: DataFrame, samplingMethod: String, minorityTypes: Array[Array[Int]]): Array[Array[String]] ={//String = {
    println("AT RUN SPARK NN")
    //trainData.show()
    var currentResults = ""
    var resultArray = Array[Array[String]]()

    var currentArray = Array[String]()

  if(samplingMethod == "smotePlus") {

      val d = trainData.select("label").distinct()
      val presentClasses = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

      for(clusterKValue <- clusterKValues) {

        val clusterResults: Map[Int, DataFrame] = presentClasses.map(x=>getClassClusters(trainData.sparkSession, x, trainData, clusterKValue)).toMap

        //val cutoff = 0.0

        for(currentMinorityTypes<-minorityTypes) {

          val t0 = System.nanoTime()
          var samples: Seq[(Int, Int, Int, Int, DenseVector)] = Seq()

          for(cutoff<-cutoffs) {
            var isValid = true
            for (classIndex <- clusterResults) {

              var currentClassCount = 0
              println("Class: " + classIndex._1)
              val predictionsCollected: Seq[(Int, Int, Int, Int, DenseVector)] = clusterResults(classIndex._1).collect().map(x => (x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).toString.toInt, x(4).asInstanceOf[DenseVector])).toSeq // FIXME - parallelize, should DF not Seq?
              val clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]] = predictionsCollected.groupBy {
                _._1
              }
              println("************************** " + clusteredData.size)

              for (currentCluster<-clusteredData) {
                val currentClusterIndex = currentCluster._1
                val currentClusterData = currentCluster._2
                val dd: Seq[Int] = currentClusterData.map(x => x._4)

                var countInCluster = 0
                for (count <- dd.groupBy(identity).mapValues(_.size)) {
                  println("\t" + count._1 + " " + count._2)
                  countInCluster += count._2
                }
                println("at check: " + countInCluster + " >= " + currentClusterData.length + " " + cutoff + " :" + currentClusterData.length * cutoff)
                if (countInCluster >= currentClusterData.length * cutoff) {
                  println("Keep cluster " + currentClusterIndex)
                  println(currentClusterData.head)
                  val values = currentClusterData.filter(x => x._1 == currentClusterIndex && (currentMinorityTypes contains x._4)).map(x => (x._1, x._2, x._3, x._4, x._5))
                  currentClassCount += values.size
                  samples = samples.union(values)
                }
                else {
                  println("Dont keep cluster " + currentClusterIndex)
                }
              }
              if (currentClassCount == 0) {
                print("------> ERROR: Class " + classIndex + " is empty")
                isValid = false
              }
              // End of class loop
            }
            println("sample size: " + samples.size)

            val combinedDf = if (isValid) {
              println("New Counts: " + samples.length)

              println("~~~~~~~ At Resample bottom~~~~~~~~```")
              import trainData.sparkSession.implicits._
              //FIXME - some could be zero if split is too small
              val bar = trainData.sparkSession.sparkContext.parallelize(samples).toDF()
              val pickedTypes = bar.withColumnRenamed("_1", "cluster")
                .withColumnRenamed("_2", "index")
                .withColumnRenamed("_3", "label")
                .withColumnRenamed("_4", "minorityType")
                .withColumnRenamed("_5", "features").sort(col("index"))
              sampleDataSmotePlus(trainData.sparkSession, pickedTypes, "smotePlus", clusterKValue, clusterResults, cutoff)

            }
            else {
              trainData.sparkSession.emptyDataFrame
            }

            var currentTypesString = "["
            for (item <- currentMinorityTypes) {
              currentTypesString += item + " "
            }
            currentTypesString = currentTypesString.substring(0, currentTypesString.length() - 1)
            currentTypesString += "]"

            val foundFirst = Try(combinedDf.first)

            foundFirst match {
              case Success(x) =>
                currentResults += samplingMethod + "," + currentTypesString + "" + clusterKValue + "," + cutoff + ","
                currentArray = resultIndex.toString +: samplingMethod +: currentTypesString +: clusterKValue.toString +: cutoff.toString +: runClassifierMinorityType(combinedDf, testData)
                resultArray = resultArray :+ currentArray
                currentResults += runClassifierMinorityType(combinedDf, testData) + "\n"
              case Failure(e) => currentResults += "," + samplingMethod + "," + currentTypesString + "\n"
            }
            resultIndex += 1
          }

          val t1 = System.nanoTime()
          print("---------------- Minorty Type: ")
          currentMinorityTypes.foreach(println)
          println("Elapsed time: " + (t1 - t0) / 1e9 + "s")
          println("----------------")

        } // End of minority type loop

      }
    } /// end if smote plus
    else {
      for(currentTypes<-minorityTypes) {
        var currentTypesString = "["
        for (item <- currentTypes) {
          currentTypesString += item + " "
        }
        currentTypesString = currentTypesString.substring(0, currentTypesString.length() - 1)
        currentTypesString += "]"

        val trainDataResampled = minorityTypeResample(trainData.sparkSession, convertFeaturesToVector(trainData), currentTypes, samplingMethod)
        val foundFirst = Try(trainDataResampled.first)

        foundFirst match {
          case Success(x) =>
            currentResults += samplingMethod + "," + currentTypesString + ",,,"
              currentArray = resultIndex.toString +: samplingMethod +: currentTypesString +: "" +: "" +: runClassifierMinorityType(trainDataResampled, testData)
            resultArray = resultArray :+ currentArray
            currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
          case Failure(e) => currentResults += "," + samplingMethod + "," + currentTypesString + "\n"
        }
        resultIndex += 1
      }
    }
    resultArray
  }

  def getSparkNNMinorityResult(x: mutable.WrappedArray[Any], index: Int, features: Any): (Int, Int, Int, mutable.WrappedArray[Double]) = {
    val wrappedArray = x

    val nearestLabels = Array[Int]()
    def getLabel(neighbor: Any): Int = {
      val index = neighbor.toString.indexOf(",")
      neighbor.toString.substring(1, index).toInt
    }

    val currentLabel = getLabel(wrappedArray(0))
    var currentCount = 0
    for(i<-1 until wrappedArray.length) {
      nearestLabels :+ getLabel(wrappedArray(i))
      if (getLabel(wrappedArray(i)) == currentLabel) {
        currentCount += 1
      }
    }
    val currentArray = features.toString.substring(1, features.toString.length-1).split(",").map(x=>x.toDouble)
    (index, currentLabel, getMinorityClassLabel(currentCount), currentArray)
  }


  def getSparkNNMinorityResultX(x: mutable.WrappedArray[Any], index: Int, features: Any): (Int, Int, Int, mutable.WrappedArray[Int]) = {
    val wrappedArray = x

    val nearestLabels = Array[Int]()
    def getLabel(neighbor: Any): Int = {
      val index = neighbor.toString.indexOf(",")
      neighbor.toString.substring(1, index).toInt
    }

    def getLabelMatch(label: Int, neighbor: Any): Int = {
      val index = neighbor.toString.indexOf(",")
      val nn = neighbor.toString.substring(1, index).toInt
      if(label == nn) {
        1
      }
      else {
        0
      }
    }

    val currentLabel = getLabel(wrappedArray(0))
    var currentCount = 0
    for(i<-1 until wrappedArray.length) {
      nearestLabels :+ getLabel(wrappedArray(i))
      if (getLabel(wrappedArray(i)) == currentLabel) {
        currentCount += 1
      }
    }
    val currentArray = features.toString.substring(1, features.toString.length-1).split(",").map(x=>x.toDouble)
    var foo = (index, currentLabel, getMinorityClassLabel(currentCount), wrappedArray.map(x=>getLabelMatch(currentLabel,x)))

    (index, currentLabel, getMinorityClassLabel(currentCount), wrappedArray.map(x=>getLabelMatch(currentLabel,x)))
  }

  def getClassClusters(spark: SparkSession, l: Int, df: DataFrame, clusterKValue: Int): (Int, DataFrame) = {
    val result = if(clusterKValue < 2) {
      val currentCase = df.filter(df("label") === l).toDF()
      (l, currentCase)
    }
    else {
      val currentCase = df.filter(df("label") === l).toDF()
      val kmeans = new KMeans().setK(clusterKValue).setSeed(1L)
      val convertedDF = convertFeaturesToVector(currentCase)
      val model2 = kmeans.fit(convertedDF)
      // Make predictions
      println("^^^^^^ cluster count: " + model2.clusterCenters.length)
      val predictions = model2.transform(convertedDF).select("prediction", "index", "label", "minorityType", "features").persist(StorageLevel.MEMORY_ONLY_SER) //.cache() //FIXME - chech cache
      (l, predictions)
    }
    result
  }


  def main(args: Array[String]) {

    val t0 = System.nanoTime()
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    if (args.length < 9) {
      println("Usage: imbalanced-spark [input file path] [kNN mode] [label column] [useHeader] [save path] [file description] [r/w mode] [r/w path] [k]")
      return
    }

    val input_file = args(0)
    val labelColumnName = args(1)

    kValue = args(2).toInt
    val useHeader = if (args.length > 3 && args(3).equals("yes")) true else false

    val savePath = args(4)
    val fileDescription = args(5)

    val rw =
      if(args.length > 6) {
        if(args(6) == "read") Array("read", args(7).toString)
        else if(args(6)=="write") {
          Array("write", args(7).toString)

        }
        else Array("","")
      }
      else { Array("","") }

    val dfDataDirectory = new File(rw(1))
    if(!dfDataDirectory.exists()) {
      dfDataDirectory.mkdirs()
    }

    args.foreach(println)

    val samplingMethods = ArrayBuffer[String]()
    for(i<- 9 until args.length) {
      samplingMethods += args(i)
    }
    println("sampling methods:")
    samplingMethods.foreach(println)

    val df = spark.read.
      option("inferSchema", true).
      option("header", useHeader).
      csv(input_file).persist(StorageLevel.MEMORY_ONLY_SER)

    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")
    var minorityTypes = Array[Array[Int]]()

    //for(i<-0 to 0) {
     for(i<-0 to 0) {
       var currentMinorityTypes = Array[Int]()
       if (0 != (i & 1)) false else {
         currentMinorityTypes = currentMinorityTypes :+ 0
       }
       if (0 != (i & 2)) false else {
         currentMinorityTypes = currentMinorityTypes :+ 1
       }
       if (0 != (i & 4)) false else {
         currentMinorityTypes = currentMinorityTypes :+ 2
       }
       if (0 != (i & 8)) false else {
         currentMinorityTypes = currentMinorityTypes :+ 3
       }
       currentMinorityTypes.foreach(println)
       minorityTypes = minorityTypes :+ currentMinorityTypes
     }

    /*************************************************************/
    import df.sparkSession.implicits._

    val enableDataScaling = true

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).persist(StorageLevel.MEMORY_ONLY_SER)

    val data: RDD[(Long, Int, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString.toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
    } else { converted }.persist(StorageLevel.MEMORY_ONLY_SER)

    df.sparkSession.sparkContext.broadcast(scaledData)

    val numSplits = 1
    val counts = scaledData.count()
    var cuts = Array[Int]()

    cuts :+= 0

    if(numSplits < 2) {
      cuts :+= (counts * 0.2).toInt
    }
    else {
      for(i <- 1 until numSplits) {
        cuts :+= ((counts / numSplits) * i).toInt
      }
    }
    cuts :+= counts.toInt

    /***********************************/
    println("xxxx " + rw(0) + " " + rw(1))
    val minorityDF =
      if(rw(0) == "read") {
        println("READ KNN")
        val readData = df.sparkSession.read.
          option("inferSchema", true).
          option("header", true).
          csv(rw(1))

        val stringToArray = udf((item: String)=>item.dropRight(1).drop(1).split(",").map(x=>x.toString.toDouble))

        readData.withColumn("features", stringToArray(col("features")))
      }
      else {
        println("CALCULATE KNN")
        val leafSize = 5
        val knn = new KNN()
          .setTopTreeSize(counts.toInt / 10)
          .setTopTreeLeafSize(leafSize)
          .setSubTreeLeafSize(2500)
          .setSeed(42L)
          .setAuxCols(Array("label", "features"))
        val model = knn.fit(scaledData).setK(kValue+1)//.setDistanceCol("distances")
        val results2: DataFrame = model.transform(scaledData)
        results2.show()
        val collected: Array[Row] = results2.select( "neighbors", "index", "features").collect()

        val minorityValueDF: Array[(Int, Int, Int, mutable.WrappedArray[Double])] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Any]],x(1),x(2))).map(x=>getSparkNNMinorityResult(x._1, x._2.toString.toInt, x._3))

        val minorityDF = scaledData.sparkSession.sparkContext.parallelize(minorityValueDF).toDF()
          .withColumnRenamed("_1","index")
          .withColumnRenamed("_2","label")
          .withColumnRenamed("_3","minorityType")
          .withColumnRenamed("_4","features").sort("index")

        minorityDF.show()

        val nElements = kValue + 1

        /** reduce  **/

        if (rw(0) == "write") {
          val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

          minorityDF.withColumn("features", stringify(col("features"))).
            repartition(1).
            write.format("com.databricks.spark.csv").
            option("header", "true").
            mode("overwrite").
            save(rw(1))
          val t1 = System.nanoTime()
          println("Elapsed time: " + (t1 - t0) / 1e9 + "s")
        }
        println("------------ Minority DF")

        minorityDF

      }.persist(StorageLevel.MEMORY_ONLY_SER)

    /***********************************/

    for(cutIndex<-0 until numSplits) {
      resultIndex = 0

      val testData = scaledData.filter(scaledData("index") < cuts(cutIndex+1) && scaledData("index") >= cuts(cutIndex)).persist()
      val trainData = scaledData.filter(scaledData("index") >= cuts(cutIndex+1) || scaledData("index") < cuts(cutIndex)).persist()

      /*************************************************************/


      val minorityDFFold = minorityDF.filter(minorityDF("index") >= cuts(cutIndex+1) || minorityDF("index") < cuts(cutIndex)).persist()

      /*************************************************************/

      val savePathString = savePath + "/" + fileDescription + "/k"
      val saveDirectory = new File(savePathString)
      if(!saveDirectory.exists()) {
        saveDirectory.mkdirs()
      }

        resultArray = Array()
        for (method <- samplingMethods) {
          val result: Array[Array[String]] = runSparkNN(minorityDFFold, testData, method, minorityTypes)
          resultArray = resultArray ++ result
        }

      val csvResults = resultArray.map(x=> x match { case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) =>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)}).toSeq
      val c = df.sparkSession.sparkContext.parallelize(csvResults).toDF
      val lookup = Map(
        "_1" -> "index",
        "_2" -> "sampling",
        "_3" -> "minorityTypes",
        "_4" -> "clusters",
        "_5" -> "cutoff",
        "_6" -> "AvAvg",
        "_7" -> "MAvG",
        "_8" -> "RecM",
        "_9" -> "Recu",
        "_10" -> "PrecM",
        "_11" -> "Precu",
        "_12" -> "FbM",
        "_13" -> "Fbu",
        "_14" -> "AvFb",
        "_15" -> "CBA"
      )

      val cols = c.columns.map(name => lookup.get(name) match {
        case Some(newname) => col(name).as(newname)
        case None => col(name)
      })

      val resultsDF = c.select(cols: _*)

      resultsDF.repartition(1).
        write.format("com.databricks.spark.csv").
        option("header", "true").
        mode("overwrite").
        save(savePath + "/" + fileDescription + "/" + cutIndex)

        println(savePath + "/" + fileDescription + "/" + cutIndex)
        trainData.unpersist()
        testData.unpersist()
        minorityDFFold.unpersist()

      val t1 = System.nanoTime()
      println("Elapsed time: " + (t1 - t0) / 1e9 + "s")
    }
  }
}
