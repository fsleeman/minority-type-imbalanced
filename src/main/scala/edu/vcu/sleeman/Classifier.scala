package edu.vcu.sleeman

import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
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
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD

import scala.collection.parallel.immutable.ParSeq
import scala.util.{Failure, Random, Success, Try}


//FIXME - turn classes back to Ints instead of Doubles
object Classifier {

  var resultIndex = 0

  val clusterKValues: Array[Int] = Array(2, 5, 10, 25, 50)
  val cutoffs: Array[Double] = Array(0.0)//, 0.05, 0.1, 0.2)
  var results = ""

  var resultArray: Array[Array[String]] = Array()

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)


  def getIndexedDF(spark: SparkSession, columns: Array[String], trainData: DataFrame): DataFrame = {
    //val keepCols = trainData.columns.map(status => if (columns.contains(status) && !status.equals("_c41")) None else status)
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

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame): Array[String]={//String ={
    import distinctClasses.sparkSession.implicits._
    //FIXME - don't calculate twice

    val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt)

    val maxLabel: Int = classLabels.max //distinctClasses.collect().map(x => x.toSeq.last.toString().toDouble.toInt).max
    val minLabel: Int = classLabels.min //distinctClasses.collect().map(x => x.toSeq.last.toString().toDouble.toInt).min
    val numberOfClasses = classLabels.length //distinctClasses.count()
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


      //???



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

        //println("CBA value: " + tp / rowColMaxValue)//maxValue(colSum, rowValueSum))
      }
      else {
        //println("CBA value NaN")
      }

      //CBA += (tp / maxValue(colSum, rowValueSum))
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



    //resultArray =  resultArray :+ Row(AvAvg, MAvG, RecM, PrecM, Recu, Precu, FbM, Fbu, AvFb, CBA) //resultArray :+ Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
    //AvAvg  + "," + MAvG + "," + RecM +"," + PrecM + "," + Recu + "," + Precu + "," + FbM + "," + Fbu + "," + AvFb + "," + CBA
    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)

  }

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount: Int): Int = {
    if (kCount >= 4) { 0 }
    else if (kCount >= 2) { 1}
    else if (kCount == 1) { 2 }
    else { 3 }
  }

  def setMinorityStatus(cls: Int, sample: (Int, Array[Int])): (Int, Int) = {
    //positive case
    if (cls == sample._1) {
      val matchingCount = sample._2.count(x => x == cls)
      (1, getMinorityClassLabel(matchingCount))
    }
    //negative case
    else {
      val matchingCount = sample._2.count(x => x != cls)
      (0, getMinorityClassLabel(matchingCount))
    }
  }

  def getMinorityClassResults(cls: Int, name: String, data: Array[NearestClassResult]) {
    val minorityStatus = data.map(x => setMinorityStatus(cls, x))

    val positiveSamples = minorityStatus.filter(x => x._1 == 1).map(x => x._2)
    val negativeSamples = minorityStatus.filter(x => x._1 != 1).map(x => x._2)

    val positiveSafeCount = positiveSamples.count(x => x == "safe")//filter(x => (x == "safe")).length
    val positiveBorderlineCount = positiveSamples.count(x => x == "borderline")//filter(x => (x == "borderline")).length
    val positiveRareCount = positiveSamples.count(x => x == "rare")//filter(x => (x == "rare")).length
    val positiveOutlierCount = positiveSamples.count(x => x == "outlier")//filter(x => (x == "outlier")).length
    val positiveResults = "Positive: " + positiveSamples.length + "\nsafe: " + positiveSafeCount + " borderline: " + positiveBorderlineCount + " rare: " + positiveRareCount + " outlier: " + positiveOutlierCount

    val negativeSafeCount = negativeSamples.count(x => x == "safe")//filter(x => (x == "safe")).length
    val negativeBorderlineCount = negativeSamples.count(x => x == "borderline")//filter(x => (x == "borderline")).length
    val negativeRareCount = negativeSamples.count(x => x == "rare")//filter(x => (x == "rare")).length
    val negativeOutlierCount = negativeSamples.count(x => x == "outlier")//filter(x => (x == "outlier")).length
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
    //for(index<-0 to x.length-1) {
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

/*  def getSmoteSample(data: Array[Array[Double]]): Unit = {
    //val convertToVector = udf((array: Seq[Double]) => {
    //  Vectors.dense(array.map(_.toDouble).toArray)
    //})

    //val results = data.map(x=>Vectors.dense(x.map(_.toDouble).toArray))
    //println("DISTANCE: " +  Math.sqrt(fastSquaredDistance(results(0), Vectors.norm(results(0), 2), results(1), Vectors.norm(results(1), 2))))
    //val foo = new VectorWithNorm(results(0))
    //val foo2 = new VectorWithNorm(results(1))
    //println(fastSquaredDistance(foo.vector, foo.norm, foo2.vector, foo2.norm))
    //println("***")
    //println(foo.fastSquaredDistance(foo2))
    //println(getSingleDistance(data(0), data(1)))

    //fastSquaredDistance
    //data.map(x=>data.map(y=>getSingleDistance(x, y)))
  }*/

  def mapRow(currentRow: Array[Any]): (Int, Array[Float]) = {
    val reverseRow = currentRow.reverse
    val cls = reverseRow.head.toString.toFloat.toInt
    val features = reverseRow.tail.map(_.toString.toFloat)
    (cls, features)
  }
/*
  def calculateMinorityClasses(spark: SparkSession, trainData: DataFrame) {
      val trainRDD = trainData.rdd.map(_.toSeq.toArray).map(x => mapRow(x))
    //trainRDD.count()

    //FIXME - is this better with broadcasting?
    val train_index = trainRDD.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    //println("******************** Class Stats *****************************")
    val countOfClasses = trainRDD.map((_, 1L)).reduceByKey(_ + _).map { case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey
    //val countResults = countOfClasses.map(x => (x._1, x._2.count(x => true)))
    //countResults.sortBy(x => x._1, true).collect().foreach(println);

    //val classes = countResults.sortBy(x => x._1).map(x => x._1)
    //println("**************************************************************")

    //println("***************** Minority Class Over/Under Resample ****************************")
    //val t0 = System.nanoTime()

    val train_data_collected = train_index.collect()
    //val tX = System.nanoTime()
    //println("Time 1: " + (tX - t0) / 1.0e9 + "s")

    val minorityData = train_index.map(x => getDistances2(x, train_data_collected)).cache()

    //val minorityDataCollected = minorityData.collect()
    val indexedLabelNames = getIndexedLabelNames(trainData)
    //val rows: Array[Row] = indexedLabelNames.collect

    /*for (cls <- classes) {
      val res = rows.filter(x => x(0) == cls)
      //println()
      //printMinorityClassResults(cls, res(0)(1).toString, minorityDataCollected.map(x => (x._2, x._3)))
    }*/
  }
*/

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

  //assume there is only one class present
  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    //val samplesToAdd = numSamples - df.count()
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
    def r = randomInts.nextInt(currentClassZipped.length) //scala.util.Random.nextInt(currentClassZipped.length)

    val rand = Array(r, r, r, r, r)
    val sampled: Array[Row] = currentClassZipped.filter(x => rand.contains(x._2)).map(x => x._1) //FIXME - issues not taking duplicates
    //FIXME - can we dump the index column?
    val values: Array[Array[Double]] = sampled.map(x=>x(3).asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)

    val r2 = Row(0, cls, -1,  Vectors.dense(ddd.map(_.toDouble)))
    r2
    //(0, cls, "", Vectors.dense(ddd.map(_.toDouble)))
    //(0, cls, "", ddd)
  }

  def smote(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {

    println("Previous DF Schema")
    df.printSchema()
    val aggregatedCounts = df.groupBy("label").agg(count("label"))

    val randomInts: Random = new scala.util.Random(42L)

    //println("************* @ smote")

    //var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0).toString().toInt //FIXME
    println(cls)

    //var smoteSamples = ArrayBuffer[Row]()
    println("numSamples: " + numSamples)
    val finaDF = if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val currentClassZipped = df.collect().zipWithIndex

      val mappedResults = spark.sparkContext.parallelize(1 to samplesToAdd.toInt).map(x => smoteSample(randomInts, currentClassZipped, cls))
      import spark.implicits._

      val mappedDF = spark.sqlContext.createDataFrame(mappedResults, df.schema)
      //val df = oldDF.sqlContext.createDataFrame(rdd, oldDF.schema)

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

    //println("************* @ smote")

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0).toString().toInt //FIXME
    println(cls)

    var smoteSamples = ArrayBuffer[Row]()
    println("numSamples: " + numSamples)
    /*var finaDF = if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val currentClassZipped = df.collect().zipWithIndex

      val mappedResults = spark.sparkContext.parallelize(1 to samplesToAdd.toInt).map(x => smoteSample(randomInts, currentClassZipped, cls))
      import spark.implicits._

      val mappedDF = spark.sqlContext.createDataFrame(mappedResults, df.schema)
      //val df = oldDF.sqlContext.createDataFrame(rdd, oldDF.schema)

      val joinedDF = df.union(mappedDF)
      println("   ~~~ NEW SCHEMA ~~~")
      joinedDF.printSchema()
      joinedDF
    }
    else {
      df
    }
    finaDF
  }*/

    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val currentClassZipped = df.collect().zipWithIndex

      //for (s <- 1 to samplesToAdd.toInt) {
      for (s <- 1 to samplesToAdd.toInt) {
        def r = randomInts.nextInt(currentClassZipped.length) //scala.util.Random.nextInt(currentClassZipped.length)

        val rand = Array(r, r, r, r, r)
        val sampled: Array[Row] = currentClassZipped.filter(x => rand.contains(x._2)).map(x => x._1) //FIXME - issues not taking duplicates
        //FIXME - can we dump the index column?
        val values: Array[Array[Double]] = sampled.map(x=>x(3).asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

        val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
        val r2 = Row(0, cls, "",  Vectors.dense(ddd.map(_.toDouble)))

        //FIXME - convert this to DenseVector
        smoteSamples += r2
      }
    }

    else {
      // we already have enough samples, skip
    }

    samples = samples ++ smoteSamples
    val currentArray = df.rdd.map(x=>Row(x(0), x(1), x(2), x(3).asInstanceOf[DenseVector])).collect()
    samples = samples ++ currentArray

    /*val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })*/

    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString, x(3).asInstanceOf[DenseVector]))//asInstanceOf[mutable.WrappedArray[Double]]))

    import df.sparkSession.implicits._
    val bar = spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
    bar2
    //println("before under: " + bar2.count() )
    //val finalDF = underSample(spark, bar2, numSamples) //FITME - check if this is the right number
    //finalDF //FIXME
  }


  def YYY(cls: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random): Row = {
    val clusterIndex = clusterIds(randomInts.nextInt(clusterIds.length))
    val sampled: Array[(Int, Int, Int, Int, DenseVector)] = Array.fill(clusterCount)(clusteredData(clusterIndex)(randomInts.nextInt(clusteredData(clusterIndex).length)))

    //FIXME - can we dump the index column?
    val values: Array[Array[Double]] = sampled.map(x=>x._5.asInstanceOf[DenseVector].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
    val r2 = Row(0, cls, 0,  Vectors.dense(ddd.map(_.toDouble)))

    //FIXME - convert this to DenseVector
    r2
  }


  def smotePlus(spark: SparkSession, df: DataFrame, numSamples: Int, predictions: DataFrame, l: Int, clusterCount: Int): DataFrame = {
       //val aggregatedCounts = df.groupBy("label").agg(count("label"))

    val randomInts: Random = new scala.util.Random(42L)

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount: Long = df.count()
        //val cls = aggregatedCounts.take(1)(0)(0).toString.toInt

        //println("~~~~~~~~~~~~~~~~~~~ cls: " + cls + " l: " + l)

    //var smoteSamples = ArrayBuffer[Row]()
    println("***^ INSIDE current count: "  + l + " " + currentCount)
    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      //val currentClassZipped = df.collect().zipWithIndex

      println("At spark Means " + samplesToAdd)

      val predictionsCollected = predictions.collect().map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).toString.toInt, x(4).asInstanceOf[DenseVector])).toSeq
      val clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]] = predictionsCollected.groupBy {_._1}
      val clusterIds = clusteredData.keySet.toSeq  //.map(x=>x._1).toSeq

      println("samplesToAdd: "  + samplesToAdd)



      val t01 = System.nanoTime()
      /*for (s <- 1 to samplesToAdd.toInt) {

        //def clusterIndex = randomInts.nextInt(clusterCount)//scala.util.Random.nextInt(clusterCount)

        //val clusterIndex = randomInts.nextInt(clusterCount)
        val clusterIndex = clusterIds(randomInts.nextInt(clusterIds.length))


        //val len = clusteredData(clusterIndex).length //FIXME - add to map
        //def getSample: (Int, Int, Int, String, DenseVector) = clusteredData(clusterIndex)(randomInts.nextInt(len) ) //(0) //clusteredData(clusterIndex)(scala.util.Random.nextInt(5))

        val sampled: Array[(Int, Int, Int, Int, DenseVector)] = Array.fill(clusterCount)(clusteredData(clusterIndex)(randomInts.nextInt(clusteredData(clusterIndex).length)))

        //FIXME - can we dump the index column?
        val values: Array[Array[Double]] = sampled.map(x=>x._5.asInstanceOf[DenseVector].toArray)//.asInstanceOf[mutable.WrappedArray[Double]].toArray)

        val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
        val r2 = Row(0, cls, 0,  Vectors.dense(ddd.map(_.toDouble)))

        //FIXME - convert this to DenseVector
        smoteSamples += r2
      }*/

      val XXX: ParSeq[Row] = (1 to samplesToAdd.toInt).par.map { _ => YYY(l: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random) }// .reduce(_ ++ _)
      samples = samples ++ XXX
      val t11 = System.nanoTime()
      println("--------- LOOP TIME: " + (t11 - t01) / 1e9 + "s")

    }
    else {
      // we already have enough samples, skip
    }

    val tX = System.nanoTime()

    //samples = samples ++ smoteSamples
    //df.show()

        //val currentArray = df.rdd.map(x=>Row(x(0), x(1), x(2), x(3).asInstanceOf[DenseVector])).collect()
    val currentArray = df.rdd.map(x=>Row(x(1), x(2), x(3), x(4).asInstanceOf[DenseVector])).collect()
    //println(currentArray.take(1)(0))

    samples = samples ++ currentArray

    /*val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })*/

    //println(samples.head)
    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).asInstanceOf[DenseVector]))//asInstanceOf[mutable.WrappedArray[Double]]))

    //import df.sparkSession.implicits._
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))  //   createDataFrame(foo)  //spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
    //bar2.show()
    //println("before under: " + bar2.count())
    val tY = System.nanoTime()
    println("--------- Combine Time: " + (tY - tX) / 1e9 + "s")
    bar2
    //val finalDF = underSample(spark, bar2, numSamples) //FIXME - check if this is the right number - something might not be right with SMOTE counts
    //finalDF //FIXME
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
      //println(df.count())
      val currentCase = df.filter(df("label") === l).toDF()
      //println("count: " + currentCase.count())
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount)
        //case "smotePlus" => smotePlus(spark, currentCase, smoteSampleCount, clusterResults(l),true)
        case _ => currentCase
      }
      //println("updated: " + filteredDF2.count())
      //dfs = dfs :+ filteredDF2
    filteredDF2
  }

  //FIXME - cutoff does not seem to be used
  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    //println("~~~~~ sampleData ~~~~~")
    //df.printSchema()
    //getCountsByClass(spark, "label", df).show()
    val d = df.select("label").distinct()
    //println("^^^^^^^ distinct classes ^^^^^^^^^")
    //d.show()
    //println("^^^^^^^ distinct classes ^^^^^^^^^")
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

    val counts = getCountsByClass(spark, "label", df)
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString.toInt

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount / 2
    //var dfs: Array[DataFrame] = Array[DataFrame]()

    /*for (l <- presentClasses) {
      println("----------> Class: " + l)
      val currentCase = df.filter(df("label") === l).toDF()
      println("count: " + currentCase.count())
      val filteredDF2 = samplingMethod match {
        case "undersample" => underSample(spark, currentCase, underSampleCount)
        case "oversample" => overSample(spark, currentCase, overSampleCount)
        case "smote" => smote(spark, currentCase, smoteSampleCount)
        //case "smotePlus" => smotePlus(spark, currentCase, smoteSampleCount, clusterResults(l),true)
        case _ => currentCase
      }
      println("updated: " + filteredDF2.count())
      dfs = dfs :+ filteredDF2
    }



    val all = dfs.reduce(_ union  _)
    println("^^^^^^^^^^^^")*/
    //df.show()

    val myDFs = presentClasses.map(x=>(x, df.filter(df("label") === x).toDF()))

    //val classDF: RDD[DataFrame] = spark.sparkContext.parallelize(presentClasses).map(x => sampleDataParallel(spark, df.filter(df("label") === x).toDF(), x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))
    val classDF: Array[DataFrame] = presentClasses.map(x => sampleDataParallel(spark, myDFs.filter(y=>y._1 == x)(0)._2, x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))

    println("Final count ")
    println(classDF.length)
    val r = classDF.reduce(_ union _)
    //println(r.count())
    r




    /*val all = dfs.reduce(_ union  _)
    println("^^^^^^^^^^^^")

    println("Total: " + all.count())
    //convertFeaturesToVector(all)
    all*/
  }

  ////  /*def minorityTypeResample(spark: SparkSession, df: DataFrame, minorityTypes: Array[String], samplingMethod: String, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {

  //FIXME - cutoff does not seem to be used
  def sampleDataSmotePlus(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterKValue: Int, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {
    //println("~~~~~ sampleData ~~~~~")
    //df.printSchema()
    //getCountsByClass(spark, "label", df).show()
    val d = df.select("label").distinct()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    d.show()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt) //Array(1,2,3,4,5,6,7)    //

    val counts = getCountsByClass(spark, "label", df)
    println("***^ COUNTS OUTSIDE")
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    //val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString.toInt

    //val overSampleCount = maxClassCount
    //val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount / 2
    var dfs: Array[DataFrame] = Array[DataFrame]()

    for (l <- presentClasses) {
      //println("----------> Class: " + l)
      val currentCase = df.filter(df("label") === l).toDF() ///FIXME - this may already been calculated
      val filteredDF2 = samplingMethod match {
        //case "undersample" => underSample(spark, currentCase, underSampleCount)
        //case "oversample" => overSample(spark, currentCase, overSampleCount)
        //case "smote" => smote(spark, currentCase, smoteSampleCount)
        case "smotePlus" => smotePlus(spark, currentCase, smoteSampleCount, clusterResults(l), l ,clusterKValue)
        case _ => currentCase
      }
      //println("updated: " + filteredDF2.count())
      dfs = dfs :+ filteredDF2
    }

    val all = dfs.reduce(_ union  _)
    //convertFeaturesToVector(all)
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

    val confusionMatrix = predictions.
      groupBy("label").
      pivot("prediction", minLabel to maxLabel).
      count().
      na.fill(0.0).
      orderBy("label")

    calculateClassifierResults(test.select("label").distinct(), confusionMatrix)
  }


  /*def runNaiveNN(df: DataFrame, samplingMethod: String, minorityTypes: Array[Array[String]], clusterResults: Map[Int,DataFrame], enableDataScaling: Boolean, rw: Array[String]): String = {
    //df.show()

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

    //trainData.show()
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
      val foundFirst = Try(trainDataResampled.first)

      println("found status: " + foundFirst.toString)
      foundFirst match {
        case Success(dummy) =>
          currentResults += samplingMethod + "," + currentTypesString + ","
          currentResults += runClassifierMinorityType(convertFeaturesToVector(trainDataResampled), testData) + "\n"

        case Failure(e) => currentResults += samplingMethod + "," + currentTypesString + '\n'
      }


    }
    currentResults
  }*/
/*
  def printSparkNNMinorityReport(df: DataFrame): Unit = {
   // println("Minority Class Types")
    val groupedDF: DataFrame = df.select("label", "minorityType").groupBy("label", "minorityType").count()
    val listOfClasses = groupedDF.select("label").distinct().select("label").collect().map(_(0)).toList

    for(currentClass<-listOfClasses) {
      var minorityTypeMap = Map[String, Int]("safe"->0, "borderline"->0, "rare"->0, "outlier"->0)

      val currentLabel = groupedDF.filter(col("label").===(currentClass)).collect()
      for(minorityType<-currentLabel) {
        minorityTypeMap += minorityType(1).toString -> minorityType(2).toString.toInt
      }
      println("Class: " + currentClass + " safe: " + minorityTypeMap("safe") + " borderline: " + minorityTypeMap("borderline") +
        " rare: " + minorityTypeMap("rare") + "  outlier: " + minorityTypeMap("outlier"))
    }
  }*/
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

              /*for (clusterIndex <- 0 until clusteredData.size) {
                val dd: Seq[Int] = clusteredData(clusterIndex).map(x => x._4)
                println("@Cluster " + clusterIndex + " length: " + clusteredData(clusterIndex).length)
                //val clusterTypes: Map[String, Seq[(Int, Int, Int, String, DenseVector)]] = clusteredData(x).groupBy {_._4}

                //val minorityTypeCounts = dd.groupBy(identity).mapValues(_.size).filter(x => minorityTypes contains x._1)
                println(dd.groupBy(identity).mapValues(_.size))

                var countInCluster = 0
                for (count <- dd.groupBy(identity).mapValues(_.size)) {
                  println("\t" + count._1 + " " + count._2)
                  countInCluster += count._2
                  //if (count._2 > clusteredData(clusterIndex).length / 10) {
                  //  println("Including: " + count._1)
                  //}
                }
                println("at check: " + countInCluster + " >= " + clusteredData(clusterIndex).length + " " + cutoff + " :" + clusteredData(clusterIndex).length * cutoff)
                if (countInCluster >= clusteredData(clusterIndex).length * cutoff) {
                  println("Keep cluster " + clusterIndex)
                  //samples = samples.union(df.filter(x=>(x)))
                  println(clusteredData(clusterIndex).head)

                  val values = clusteredData(clusterIndex).filter(x => x._1 == clusterIndex && (currentMinorityTypes contains x._4)).map(x => (x._1, x._2, x._3, x._4, x._5))
                  println("values size: " + clusteredData(clusterIndex).count(x => x._1 == clusterIndex && (currentMinorityTypes contains x._4)))//filter(x => x._1 == clusterIndex && (currentMinorityTypes contains x._4)).size)
                  currentClassCount += values.size
                  samples = samples.union(values)
                }
                else {
                  println("Dont keep cluster " + clusterIndex)
                }
              }*/

              for (currentCluster<-clusteredData) {
                //val foo: (Int, Seq[(Int, Int, Int, Int, DenseVector)]) = currentCluster
                val currentClusterIndex = currentCluster._1
                val currentClusterData = currentCluster._2
                val dd: Seq[Int] = currentClusterData.map(x => x._4)

                //println("@Cluster " + clusterIndex + " length: " + clusteredData(clusterIndex).length)
                //val clusterTypes: Map[String, Seq[(Int, Int, Int, String, DenseVector)]] = clusteredData(x).groupBy {_._4}

                //val minorityTypeCounts = dd.groupBy(identity).mapValues(_.size).filter(x => minorityTypes contains x._1)
                //println(dd.groupBy(identity).mapValues(_.size))

                var countInCluster = 0
                for (count <- dd.groupBy(identity).mapValues(_.size)) {
                  println("\t" + count._1 + " " + count._2)
                  countInCluster += count._2
                  //if (count._2 > clusteredData(clusterIndex).length / 10) {
                  //  println("Including: " + count._1)
                  //}
                }
                println("at check: " + countInCluster + " >= " + currentClusterData.length + " " + cutoff + " :" + currentClusterData.length * cutoff)
                if (countInCluster >= currentClusterData.length * cutoff) {
                  println("Keep cluster " + currentClusterIndex)
                  //samples = samples.union(df.filter(x=>(x)))
                  println(currentClusterData.head)
                  //FIXME - probably don't need to check the index, should be determined by current loop
                  val values = currentClusterData.filter(x => x._1 == currentClusterIndex && (currentMinorityTypes contains x._4)).map(x => (x._1, x._2, x._3, x._4, x._5))
                  //println("values size: " + currentClusterData.count(x => x._1 == currentClusterIndex && (currentMinorityTypes contains x._4)))//filter(x => x._1 == clusterIndex && (currentMinorityTypes contains x._4)).size)
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

                //trainData.sparkSession.emptyDataFrame //FIXME - could exit sooner
              }
              // End of class loop
            }
            println("sample size: " + samples.size)
            // assert(false)

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

              //println("Picked samples count: " + pickedTypes.count())
              //pickedTypes.show()
              //assert(false)
              //FIXME - avoid passing spark as parameter?
              //val combinedDf = sampleData(spark, pickedTypes, "smote")
              //combinedDf
              //sampleData(trainData.sparkSession, pickedTypes, "smote")
              sampleDataSmotePlus(trainData.sparkSession, pickedTypes, "smotePlus", clusterKValue, clusterResults, cutoff)

            }
            else {
              trainData.sparkSession.emptyDataFrame
            }


            //for(currentTypes<-currentMinorityTypes) {
            var currentTypesString = "["
            for (item <- currentMinorityTypes) {
              currentTypesString += item + " "
            }
            currentTypesString = currentTypesString.substring(0, currentTypesString.length() - 1)
            currentTypesString += "]"

            //val trainDataResampled = minorityTypeResample(combinedDf.sparkSession, convertFeaturesToVector(combinedDf), currentMinorityTypes, samplingMethod)
            //val trainDataResampled = minorityTypeResample(combinedDf.sparkSession, combinedDf, currentMinorityTypes, samplingMethod)

            //val trainDataResampled = minorityTypeResample(trainData.sparkSession, trainData, currentTypes, samplingMethod, clusterResults, 0.0)
            //trainDataResampled.printSchema()
            val foundFirst = Try(combinedDf.first)

            foundFirst match {
              case Success(x) =>
                currentResults += samplingMethod + "," + currentTypesString + "" + clusterKValue + "," + cutoff + ","
                //Array(samplingMethod, currentTypesString , clusterKValue, cutoff) ++
                currentArray = resultIndex.toString +: samplingMethod +: currentTypesString +: clusterKValue.toString +: cutoff.toString +: runClassifierMinorityType(combinedDf, testData)
                resultArray = resultArray :+ currentArray
                //currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
                currentResults += runClassifierMinorityType(combinedDf, testData) + "\n"
              case Failure(e) => currentResults += "," + samplingMethod + "," + currentTypesString + "\n"
            }
            resultIndex += 1
            // }
            //println("Final count: " + combinedDf.count())
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
        //val trainDataResampled = minorityTypeResample(trainData.sparkSession, trainData, currentTypes, samplingMethod, clusterResults, 0.0)
        //trainDataResampled.printSchema()
        val foundFirst = Try(trainDataResampled.first)

        foundFirst match {
          case Success(x) =>
            currentResults += samplingMethod + "," + currentTypesString + ",,,"
            //Array(samplingMethod, currentTypesString , "", 0) ++
              currentArray = resultIndex.toString +: samplingMethod +: currentTypesString +: "" +: "" +: runClassifierMinorityType(trainDataResampled, testData)
            resultArray = resultArray :+ currentArray
            //currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
            currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
          case Failure(e) => currentResults += "," + samplingMethod + "," + currentTypesString + "\n"
        }
        resultIndex += 1
      }
    }


/*
    if(samplingMethod == "smotePlus") {
      for(clusterValue <- clusterKValues) {

        //generate clusters



        for(cutoff<-cutoffs) {
          for(currentTypes<-minorityTypes) {
            var currentTypesString = "["
            for (item <- currentTypes) {
              currentTypesString += item + " "
            }
            currentTypesString = currentTypesString.substring(0, currentTypesString.length()-1)
            currentTypesString += "]"

            val trainDataResampled = minorityTypeResample(trainData.sparkSession, convertFeaturesToVector(trainData), currentTypes, samplingMethod, clusterValue, cutoff) //FIXME
            //val trainDataResampled = minorityTypeResample(trainData.sparkSession, trainData, currentTypes, samplingMethod, clusterResults, 0.0)
            //trainDataResampled.printSchema()
            val foundFirst = Try(trainDataResampled.first)

            foundFirst match {
              case Success(x) => //do stuff with the dataframe
                currentResults += samplingMethod + "," + currentTypesString + "," + clusterValue + "," + cutoff + ","
                //currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
                currentResults += runClassifierMinorityType(trainDataResampled, testData) + "\n"
              case Failure(e) => currentResults += samplingMethod + "," + currentTypesString + "," + clusterValue + "," + cutoff + "," + "\n"
              // dataframe is empty; do other stuff
              //e.getMessage will return the exception message
            }
          }
        }
      }
    }
    else {
      //FIXME
      for(currentTypes<-minorityTypes) {
        var currentTypesString = "["
        for (item <- currentTypes) {
          currentTypesString += item + " "
        }
        currentTypesString = currentTypesString.substring(0, currentTypesString.length()-1)

        currentTypesString += "]"






      }
    }


*/

    //currentResults
  //val foo: Array[String] = currentResults.split(",")
  //println("at foo")
  //foo.foreach(println)


  //println("**at array")
  //currentArray.foreach(println)
  //println("**end")
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
    (index, currentLabel, getMinorityClassLabel(currentCount), currentArray)//features.toString().asInstanceOf[mutable.WrappedArray[Double]])
  }


  def getClassClusters(spark: SparkSession, l: Int, df: DataFrame, clusterKValue: Int): (Int, DataFrame) = {
    //spark.sqlContext.emptyDataFrame
    val result = if(clusterKValue < 2) {
      val currentCase = df.filter(df("label") === l).toDF()
      //val convertedDF = convertFeaturesToVector(currentCase)
      (l, currentCase)
    }
    else {
      val currentCase = df.filter(df("label") === l).toDF()
      val kmeans = new KMeans().setK(clusterKValue).setSeed(1L)
      val convertedDF = convertFeaturesToVector(currentCase)
      val model2 = kmeans.fit(convertedDF)
      // Make predictions
      println("^^^^^^ cluster count: " + model2.clusterCenters.length)
      val predictions = model2.transform(convertedDF).select("prediction", "index", "label", "minorityType", "features").persist()  //.cache() //FIXME - chech cache
      (l, predictions)
    }
    result
  }


  def main(args: Array[String]) {

    val t0 = System.nanoTime()
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    //print(args.length)
    //for(x<- args) {
      //print(x + '\n')
    //}
    if (args.length < 9) {
      println("Usage: imbalanced-spark [input file path] [kNN mode] [label column] [useHeader] [save path] [file description] [r/w mode] [r/w path] [k]")
      return
    }
    //println("*****")
    //print("Args:")
    // args.foreach(println)
    // println("*****")

    val input_file = args(0)
    val labelColumnName = args(1)

    val mode = args(2)
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

    //val clusterCountXX = args(8).toInt
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
      csv(input_file).persist()

    //val df = df1//.repartition(8)
    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)

    //val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)
    var minorityTypes = Array[Array[Int]]()

    //for(i<-0 to 0) {
     for(i<-0 to 14) {
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
       //currentMinorityTypes = Array[String]("rare", "outlier")//FIXME
       minorityTypes = minorityTypes :+ currentMinorityTypes
     }

    /*************************************************************/
    import df.sparkSession.implicits._

    val enableDataScaling = true

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString.toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
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
    } else { converted }.cache()

    df.sparkSession.sparkContext.broadcast(scaledData)

    val numSplits = 5
    val counts = scaledData.count()
    var cuts = Array[Int]()

    cuts :+= 0

    if(numSplits < 2) {
      cuts :+= (counts * 0.2).toInt
    }
    else {
      for(i <- 1 until numSplits) {
        cuts :+= ((counts / numSplits) * i).toInt
        //cuts :+= 2000
      }
    }
    cuts :+= counts.toInt
    //cuts.foreach(println)

    /*println("***")
    for(cutIndex<-0 to numSplits-1) {
      println(cutIndex)
    }
    println("***")*/

    //for(cutIndex<-0 to cuts.length-2) {

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
          .setSubTreeLeafSize(leafSize)
          .setSeed(42L)
          .setAuxCols(Array("label", "features"))
        val model = knn.fit(scaledData).setK(6)//.setDistanceCol("distances")
        val results2: DataFrame = model.transform(scaledData)

        val collected: Array[Row] = results2.select( "neighbors", "index", "features").collect()
        val minorityValueDF: Array[(Int, Int, Int, mutable.WrappedArray[Double])] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Any]],x(1),x(2))).map(x=>getSparkNNMinorityResult(x._1, x._2.toString.toInt, x._3))

        val minorityDF = scaledData.sparkSession.sparkContext.parallelize(minorityValueDF).toDF()
          .withColumnRenamed("_1","index")
          .withColumnRenamed("_2","label")
          .withColumnRenamed("_3","minorityType")
          .withColumnRenamed("_4","features").sort("index")
        //printSparkNNMinorityReport(minorityDF)

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
        //minorityDF.show()
        return
        minorityDF

      }.persist()

    /***********************************/

    for(cutIndex<-0 until numSplits) {
      //println(cuts(cutIndex) + " " + (cuts(cutIndex+1)))
      resultIndex = 0

      val testData = scaledData.filter(scaledData("index") < cuts(cutIndex+1) && scaledData("index") >= cuts(cutIndex)).persist()
      val trainData = scaledData.filter(scaledData("index") >= cuts(cutIndex+1) || scaledData("index") < cuts(cutIndex)).persist()

      //println("train: " + trainData.count())
      //println("test: " + testData.count())

      //val Array(trainData, testData) = scaledData.randomSplit(Array(0.8, 0.2),42L)
      //println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      //trainData.printSchema()

      /*************************************************************/


      val minorityDFFold = minorityDF.filter(minorityDF("index") >= cuts(cutIndex+1) || minorityDF("index") < cuts(cutIndex)).persist()

      /*************************************************************/


      //val samplingMethods = Array("none")//, "smotePlus")

      //val d = trainData.select("label").distinct()
      //val presentClasses = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString().toInt)


      //val clusterResults: Map[Int, DataFrame] = presentClasses.map(x=>getClassClusters(spark, x, minorityDFFold)).toMap
     // val clusterResults: Map[Int, DataFrame] = presentClasses.map(x=>getClassClusters(spark, x, trainData)).toMap


      val savePathString = savePath + "/" + fileDescription + "/k" // + clusterCount.toString
      val saveDirectory = new File(savePathString)
      if(!saveDirectory.exists()) {
        saveDirectory.mkdirs()
      }

      //val writer = new PrintWriter(new File(savePathString + "/" + fileDescription + "_k" + clusterCount.toString + "_" + cutIndex.toString + ".csv"))
      //println(savePathString + "/" + fileDescription + "_k" + "_" + cutIndex.toString + ".csv")

      //val writer = new PrintWriter(new File(savePathString + "/" + fileDescription + "_k" + "_" + cutIndex.toString + ".csv"))
        resultArray = Array()
        //writer.write("sampling,minorityTypes,clusters,cutoff,AvAvg,MAvG,RecM,PrecM,Recu,Precu,FbM,Fbu,AvFb,CBA\n")
        for (method <- samplingMethods) {
          //println("method: " + method)

          val result: Array[Array[String]] = runSparkNN(minorityDFFold, testData, method, minorityTypes)
          //result.foreach(println)
          //resultArray = resultArray :+ Row.fromSeq(result)


          resultArray = resultArray ++ result


          //resultIndex += 1
          //resultArray = Array("sampling,minorityTypes,clusters,cutoff")
          //resultArray = resultArray :+ result

          //writer.write(runSparkNN(preppedDataUpdated, method, minorityTypes, true))
          //writer.write(runSparkNN(minorityDFFold, testData, method, minorityTypes))
         // writer.flush()
        }
        //writer.close()


      val csvResults = resultArray.map(x=> x match { case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) =>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)}).toSeq


      //val gg: Seq[(String, String, String, String, String, String, String, String, String, String, String, String, String, String)] = Seq(("None","[0 1 2 3]","","","0.547","0.0","0.044919786096256686","NaN","0.22459893048128343","0.28","0.0","0.26683608640406614","0.11699164345403901","0.044919786096256686"),
      //  ("None","[1 2 3]","","","0.545","0.0","0.0378698224852071","NaN","0.1893491124260355","0.21333333333333335","0.0","0.20806241872561768","0.093841642228739","0.0378698224852071"))

      //val d=df.sparkSession.sparkContext.parallelize(Seq(("None","[0 1 2 3]","","","0.547","0.0","0.044919786096256686","NaN","0.22459893048128343","0.28","0.0","0.26683608640406614","0.11699164345403901","0.044919786096256686"),
      //("None","[1 2 3]","","","0.545","0.0","0.0378698224852071","NaN","0.1893491124260355","0.21333333333333335","0.0","0.20806241872561768","0.093841642228739","0.0378698224852071"))).toDF
      //d.show()


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

    //println("length: " + resultArray.length)




    /*for(i<-resultArray) {

      def toTuple5 = i match { case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)}
      for(j<-i) {
        print(j + ",")
      }
      println("\n")
    }*/




/*
    val xxx = c.withColumnRenamed("_1", "sampling").withColumnRenamed("_2", "minorityTypes").withColumnRenamed("_3", "clusters")

      xxx.show()

    c.repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save("/tmp/foo")

*/
    /*val x: RDD[Row] = df.sparkSession.sparkContext.parallelize(zz)


    val rdd = df.sparkSession.sparkContext.parallelize(resultArray)
    val xxxxxx = rdd.take(0).length
    println(xxxxxx.toString)*/
/*
    rdd.collect().foreach(println)

*/
    /*

    val df2 = df.sparkSession.sqlContext.createDataFrame(
        rdd,
      StructType(Seq(StructField("sampling", StringType),

        StructField("minorityTypes", StringType, true),
        StructField("clusters", StringType, true),
        StructField("cutoff", StringType, true),

        StructField("AvAvg", StringType, true),
        StructField("MAvG", StringType, true),
        StructField("RecM", StringType, true),
        StructField("Recu", StringType, true),
        StructField("PrecM", StringType, true),
        StructField("Precu", StringType, true),
        StructField("FbM", StringType, true),
        StructField("Fbu", StringType, true),
        StructField("AvFb", StringType, true),
        StructField("CBA", StringType, true))

        //StructType(Seq(StructField("arr", ArrayType(StringType, false), false)
      ))
    df2.show()

    val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

    df2.
      repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save("/tmp/foo")

    println(rdd)



    val rdd2 = df.sparkSession.sparkContext.parallelize(
      Seq(
        Row.fromSeq(Array("x", 2.0, 1.0, 2.1, 5.4).toSeq),
        Row.fromSeq(Array("y", 1.5, 0.5, 0.9, 3.7).toSeq),
        Row.fromSeq(Array("z", 8.0, 2.9, 9.1, 2.5).toSeq)
      )
    )


    val schema = new StructType()
      .add(StructField("foo", StringType, true))
      .add(StructField("id", DoubleType, true))
      .add(StructField("val1", DoubleType, true))
      .add(StructField("val2", DoubleType, true))

    val dfWithoutSchema = df.sparkSession.createDataFrame(rdd2, schema)
    dfWithoutSchema.show()
*/

    /*val schema = new StructType()
      .add(StructField("sampling", StringType))
      .add(StructField("sampling", StringType, true))
      .add(StructField("minorityTypes", StringType, true))
      .add(StructField("clusters", StringType, true))
      .add(StructField("cutoff", StringType, true))

      .add(StructField("AvAvg", StringType, true))
      .add(StructField("MAvG", StringType, true))
      .add(StructField("RecM", StringType, true))
      .add(StructField("Recu", StringType, true))
      .add(StructField("PrecM", StringType, true))
      .add(StructField("Precu", StringType, true))
      .add(StructField("FbM", StringType, true))
      .add(StructField("Fbu", StringType, true))
      .add(StructField("AvFb", StringType, true))
      .add(StructField("CBA", StringType, true))
*/
    //val resultDF = spark.createDataFrame(x)
    //resultDF.show()

    //val xx = df.sparkSession.sparkContext.parallelize(resultArray.toSeq)
    //df.sparkSession.createDataFrame(xx, schema).show()

    //val dfWithSchema = df.sparkSession.createDataset(x).toDF("id", "vals")
    //val y = df.sparkSession.createDataFrame(x)

    //x.show()



  }
}
