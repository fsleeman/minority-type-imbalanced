package edu.vcu.sleeman

import org.apache.spark._
import java.io._

import scala.util.Random
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.rdd._
import scala.math._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql._


object MinorityType {
  type Element = (Long, (Int, Array[Float]))
  type ElementNI = (Int, Array[Float])
  type MinoriyElement = (Element, String)
  type MinoriyElementNI = (Int, Array[Float], String)
  type DistanceResult = (Float, Int)
  type DistanceArray = (Long, Array[DistanceResult])
  type MinorityResult = (Int, (Array[Float], String))
  type MinorityResult2 = (Int, (Array[Float], Int))
  type MinorityGroupTypeResult = (Int, String)

  // Read input csv file and translate to (case, [data points])
  def parseLine(line:String)= {
    val fields = line.split(",")
    val reversed = fields.reverse
    (reversed.head.toInt, reversed.takeRight(10).map(x=>x.toFloat))
  }

  def getDistanceValue(train:Element, test:Element) : DistanceResult={
    if(train._1 == test._1) {
      return (Float.MaxValue, train._2._1)
    }
    else {
      var zipped = test._2._2.zip(train._2._2)
      var result = zipped.map({case(x,y)=>(x-y)*(x-y)})
      return (sqrt(result.sum).toFloat, train._2._1)
    }
  }

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount:Int):String ={
    if(kCount >= 4) {
      return "safe"
    }
    else if(kCount >= 2) {
      return "borderline"
    }
    else if(kCount == 1) {
      return "rare"
    }
    else {
      return "outlier"
    }
  }

  //def getNewElement(cls:Float, data:RDD[Array[Float]]): ElementNI={
  def getNewElement(data:Array[MinoriyElementNI]): ElementNI={
    val dataValues = data.map(x=>x._2)
    val elementsSampled = dataValues.transpose.map(x=>x.sum/5)
    return (data(0)._1, elementsSampled.toArray)
  }

  def getNewElementC(cls:Int, data:RDD[Array[Float]]): ElementNI={
    val dataSeq = data.collect().toSeq
    val elementsSampled = dataSeq.map(x=>x.toSeq).transpose.map(x=>x.sum/5)
    return (cls, elementsSampled.toArray)
  }

  def getDistances(current:Element, train:Array[Element]): MinorityResult ={
    val result = train.map(x => getDistanceValue(x, current)).sortBy(x=>x._1).take(5)
    val cls = current._2._1
    val sum = result.filter(x=>(x._2==cls)).length
    return (current._2._1, (current._2._2, getMinorityClassLabel(sum)))
  }

  def getMinorityLabels(sample: Element, data:Array[Element]): MinorityResult ={
    val currentClass = sample._2._1
    val samples = data.filter(x=>(x._2._1==currentClass))/// && x._1 != sample._1))
    val result = samples.map(x => getDistanceValue(x, sample)).take(5)
    val sum = result.filter(x=>(x._2==currentClass)).length
    return (sample._2._1, (sample._2._2, getMinorityClassLabel(sum)))
  }


  def getMinorityDistance(sample: Element, data:Array[Element]): Float ={
    val currentClass = sample._2._1
    val samples = data.filter(x=>(x._2._1==currentClass && x._1 != sample._1))
    val result = samples.map(x => getDistanceValue(x, sample)).sortBy(x=>x._1).take(5)
    val sum = result.filter(x=>(x._2==currentClass)).length
    return result.head._1
  }

  def getIndicies(maxIndex:Int):Array[Int] ={
    val r = scala.util.Random
    val indices = (1 to (maxIndex)).map(x=>r.nextInt(maxIndex)).toArray
    return indices
  }

  def resampleData(data:RDD[MinoriyElementNI], sc:SparkContext): ListBuffer[ElementNI] ={
    val countOfClasses = data.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v, j), cnt) => (k, (v,j,cnt)) }.groupByKey
    val maxCount = countOfClasses.map(x=>x._2.count(x=>true)).collect().max
    val countPerClass = maxCount / 2;
    val listOfClasses = countOfClasses.map(x=>x._1).collect().toSeq

    var insertItems = new ListBuffer[ElementNI]()

    val borderlineData = data.filter(x=>(x._3=="safe"||x._3=="borderline"))
    // Initial example from: https://stackoverflow.com/questions/35763284/spark-group-by-key-then-count-by-value
    val countOfBorderlineData = borderlineData.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v, j), cnt) => (k, (v,j,cnt)) }.groupByKey
    val maxCountB = countOfClasses.map(x=>x._2.count(x=>true)).collect().max
    val countPerClassB = maxCountB / 2;
    val listOfClassesB = borderlineData.map(x=>x._1).collect().toSeq

    val minorityCountMax = countOfBorderlineData.map(x=>x._2.toSeq.length).max()

    for(cls<-countOfBorderlineData.collect()) {
      if(cls._2.toSeq.length > countPerClass) {
        //undersample
        val undersampled = data.filter(x=>(x._1==cls._1 && (x._3=="safe"||x._3=="borderline"))).takeSample(false, countPerClass, System.nanoTime.toInt).map(x=>(x._1, x._2))
        for(x<-undersampled) {
          insertItems += x
        }
      }
      else {
        val existingData = data.filter(x=>x._1==cls._1 && (x._3=="safe"||x._3=="borderline"))//.map(x=>(x._1, x._2)).collect()
        val existingDataMapped = existingData.map(x=>(x._1, x._2)).collect()
        for(element<-existingDataMapped) {
          insertItems += element
        }
        val selectedData = existingData.collect()
        val indicies = sc.parallelize((1 to (minorityCountMax-existingData.count().toInt)).map(x=>getIndicies(selectedData.length.toInt)))
        val results = indicies.map(x=>x.map(x=>selectedData(x))).map(x=>getNewElement(x)).collect()
        for(x<-results) {
          insertItems += x
        }
      }
    }
    return insertItems
  }

  def simpleOverSampling(x: (Int, Iterable[Array[Float]]), maxCount: Int): (Int, Array[Array[Float]]) ={
    val currentClass = x._1
    val foo = x._2.toArray
    val rng = new Random(System.currentTimeMillis)

    val elementsToAdd = maxCount - foo.length

    var samples = ArrayBuffer[Array[Float]]()
    for(i<-1 to elementsToAdd) {
      val xx = Array.fill(5)(foo(rng.nextInt(foo.size)))

      val elementSampled = xx.transpose.map(x=>x.sum/5)
      samples += elementSampled
    }
    val bar = samples.toArray

    (currentClass, foo ++ bar)
  }

  def getMinorityReport(data:(Int, Iterable[String])): MinorityGroupTypeResult ={
    val currentClass = data._1
    val values = data._2.toArray
    val safe = values.filter(x=>(x=="safe"))
    val borderline = values.filter(x=>(x=="borderline"))
    val rare = values.filter(x=>(x=="rare"))
    val outlier = values.filter(x=>(x=="outlier"))

    val safeCount = safe.length
    val borderlineCount = borderline.length
    val rareCount = rare.length
    val outlierCount = outlier.length
    val str = "safe: "+  safeCount  + " borderline: " + borderlineCount  + " rare: " + rareCount + " outlier: " +  outlierCount
    (data._1, str)
  }

  def getMinorityClassData(df: DataFrame) {
    df.show()
    val t0 = System.nanoTime()
    val train_index = df.rdd.zipWithIndex().map({case(x,y)=>(y,x)}).cache()

    val train_data = train_index.map({r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped = array.tail.map(_.toString().toFloat)
      (r._1, (cls, rowMapped))
    })
    //println(train_data.count())
    //train_data.take(20).foreach(println)
    val train_data_collected: Array[(Long, (Int, Array[Float]))] = train_data.collect()
    //println(train_data_collected(0))
    //return

    //println("Time 1: " + (tX - t0)/1.0e9 + "s")
    val minorityData = train_data.map(x=>getDistances(x, train_data_collected)).cache()  //FIXME try not caching this?
    println(minorityData.count())
    val minorityDataGrouped = minorityData.map(x=>(x._1, x._2._2)).groupByKey().cache()//partitionBy(new HashPartitioner(numPartitions)).cache()//.groupByKey()//.
    println("minorityDataSize:" +  minorityDataGrouped.count())
    println("**** Minority Class Counts ****")
    val results = minorityDataGrouped.map(x=>getMinorityReport(x)).cache()
    val t1 = System.nanoTime()
    results.sortBy(x => x._1, true).collect().foreach(println)
    println("Elapsed time: " + (t1 - t0)/1.0e9 + "s")
  }

  def getMinorityTypeStatus(df: DataFrame) {

    val train_index = df.rdd.zipWithIndex().map({case(x,y)=>(y,x)}).cache()

    val train_data = train_index.map({r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped = array.tail.map(_.toString().toFloat)
      (r._1, (cls, rowMapped))
    })

    val train_data_collected: Array[(Long, (Int, Array[Float]))] = train_data.collect()

    val countOfClasses = train_data.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey
    val countOfClassesTest = train_data.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey

    val maxCount = countOfClasses.map(x=>x._2.count(x=>true)).collect().max
    val listOfClasses = countOfClasses.map(x=>x._1).collect().toSeq

    val countResults = countOfClasses.map(x=>(x._1, x._2.count(x=>true)))

    countResults.sortBy(x => x._1, true).collect().foreach(println);
    val t0 = System.nanoTime()

   // val train_data_collected = train_index.collect()///trainIndexBroadcast.value;  //train_index.collect()

    val tX = System.nanoTime()
    println("Time 1: " + (tX - t0)/1.0e9 + "s")

    val minorityData = train_data.map(x=>getDistances(x, train_data_collected)).cache()  //FIXME try not caching this?
    val minorityDataGrouped = minorityData.map(x=>(x._1, x._2._2)).groupByKey().cache()

    println("minorityDataSize:" +  minorityDataGrouped.count())
    println("**** Minority Class Counts ****")

    val results: RDD[(Int, String)] = minorityDataGrouped.map(x=>getMinorityReport(x)).cache()
    val t1 = System.nanoTime()
    results.sortBy(x => x._1, true).collect().foreach(println);
    println("Elapsed time: " + (t1 - t0)/1.0e9 + "s")

    //FIXME -return indicies per class/minority type

  }


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val train_file = args(0)
    val numCores = args(1).toInt
    val numPartitions = numCores * 1

    println("file: "  +  train_file)
    println("cores: " + numCores.toString())
    val conf = new SparkConf()
      .setAppName("KNN")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    println(sc.getConf.getAll.mkString("\n"))

    val input_lines = sc.textFile(train_file, numPartitions)

    val input_data = input_lines.map(parseLine).cache()
    println("elements" + input_data.count())
    val splits = input_data.randomSplit(Array(1.0, 0.0), seed = 11L)
    val train_data = splits(0).cache()//.partitionBy(new HashPartitioner(numPartitions)).cache()
    val test_data = splits(1).cache()

    val train_index = train_data.zipWithIndex().map({case(x,y)=>(y,x)}).cache()
    println(train_index.take(1))

    val trainIndexBroadcast = sc.broadcast(train_index.collect())


    println("******************** Base Classifier *****************************")
    val countOfClasses = train_data.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey
    val countOfClassesTest = test_data.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey

    val maxCount = countOfClasses.map(x=>x._2.count(x=>true)).collect().max
    val listOfClasses = countOfClasses.map(x=>x._1).collect().toSeq

    val countResults = countOfClasses.map(x=>(x._1, x._2.count(x=>true)))
    println("Class counts training")

    countResults.sortBy(x => x._1, true).collect().foreach(println);

    //println("**********")
    //val countResultsTest = countOfClassesTest.map(x=>(x._1, x._2.count(x=>true)))
    //println("Class counts test")
    //for(x<-countResultsTest) {
    //  println(x._1.toString() + " " + (x._2.toString()))
    // }


    println("***************** Minority Class Over/Under Resample ****************************")
    val t0 = System.nanoTime()

    val train_data_collected = train_index.collect()///trainIndexBroadcast.value;  //train_index.collect()

    val tX = System.nanoTime()
    println("Time 1: " + (tX - t0)/1.0e9 + "s")

    //val train_data_collected = sc.broadcast(train_index.collect()) 
    // println(train_data_collected.value.length)
    //val minorityData = train_index.map(x=>getMinorityLabels(x, train_data_collected))
    val minorityData = train_index.map(x=>getDistances(x, train_data_collected)).cache()  //FIXME try not caching this?
    //for(x<-minorityData) {
    //println(x._1, x._2._1, x._2._2)
    //}
    println(minorityData.count())




    //
    val minorityDataGrouped = minorityData.map(x=>(x._1, x._2._2)).groupByKey().cache()//partitionBy(new HashPartitioner(numPartitions)).cache()//.groupByKey()//.
    //
    //println(minorityDataGrouped)
    //  type MinorityResult = (Int, (Array[Float], String))
    //
    println("minorityDataSize:" +  minorityDataGrouped.count())
    //  type MinorityResult = (Int, (Array[Float], String))
    // Minority class count
    println("**** Minority Class Counts ****")

    val results: RDD[(Int, String)] = minorityDataGrouped.map(x=>getMinorityReport(x)).cache()
    val t1 = System.nanoTime()
    results.sortBy(x => x._1, true).collect().foreach(println);
    println("Elapsed time: " + (t1 - t0)/1.0e9 + "s")
    return
    //for(x<-results) {
    //println(x)

    //for(y<-x._2) {
    //  println("\n" + y)
    //  }
    //}


  }
}
  