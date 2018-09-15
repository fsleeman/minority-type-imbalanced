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
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.reflect.ClassTag


object MinorityType {
  type Element = (Long, (Int, Array[Double]))
  type ElementNI = (Int, Array[Double])
  type MinoriyElement = (Element, String)
  type MinoriyElementNI = (Int, Array[Double], String)
  type DistanceResult = (Double, Int)
  type DistanceArray = (Long, Array[DistanceResult])
  type MinorityResult = (Int, (Array[Double], String))
  type MinorityResultIndexed = (Long, Int, String, Array[Double])

  type MinorityResult2 = (Int, (Array[Double], Int))
  type MinorityGroupTypeResult = (Int, String)


  def getMinorityReport(df: DataFrame)={//: MinorityGroupTypeResult = {
    val safeCount = df.filter(df("_3")==="safe").count()
    val borderlineCount = df.filter(df("_3")==="borderline").count()
    val rareCount = df.filter(df("_3")==="rare").count()
    val outlierCount = df.filter(df("_3")==="outlier").count()

    val str = "safe: "+  safeCount  + " borderline: " + borderlineCount  + " rare: " + rareCount + " outlier: " +  outlierCount
    println(df.take(1)(0)(1) + ":" + str)
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

  def getDistanceValue(train:Element, test:Element) : DistanceResult={
    if(train._1 == test._1) {
      return (Double.MaxValue, train._2._1)
    }
    else {
      var zipped = test._2._2.zip(train._2._2)
      var result = zipped.map({case(x,y)=>(x-y)*(x-y)})
      return (sqrt(result.sum).toDouble, train._2._1)
    }
  }

  def getDistances(current:Element, train:Array[Element]): MinorityResultIndexed ={
    val result = train.map(x => getDistanceValue(x, current)).sortBy(x=>x._1).take(5)
    val cls = current._2._1
    val sum = result.filter(x=>(x._2==cls)).length
    return (current._1, current._2._1, getMinorityClassLabel(sum), current._2._2)
  }

  def getMinorityDistance(sample: Element, data:Array[Element]): Double ={
    val currentClass = sample._2._1
    val samples = data.filter(x=>(x._2._1==currentClass && x._1 != sample._1))
    val result = samples.map(x => getDistanceValue(x, sample)).sortBy(x=>x._1).take(5)
    val sum = result.filter(x=>(x._2==currentClass)).length
    return result.head._1
  }

  def getMinorityTypeStatus(df: DataFrame): DataFrame = {
    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val train_data = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString().toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString().toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, (cls, rowMapped.reverse))
    })

    val train_data_collected: Array[(Long, (Int, Array[Double]))] = train_data.collect()

    //FIXME - are the index values needed or will the order always be in the right direction?
    val minorityData: RDD[(Long, Int, String, Array[Double])] = train_data.map(x => getDistances(x, train_data_collected)).cache() //FIXME try not caching this?
    //FIXME -return indicies per class/minority type

    val sqlContext = df.sqlContext
    import sqlContext.implicits._

    val results: DataFrame = minorityData.toDF()

    results.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
  }

  def getMinorityTypeStatus2(df: DataFrame): DataFrame = {
    import df.sparkSession.implicits._

    val train_data = df.rdd.map({r=>
      val dataString = r(2).toString()
      val array: Array[Double] = dataString.substring(1, dataString.length-1).split(',').map(_.toDouble)
      (r(0).toString().toLong, (r(1).toString().toInt, array))
      })

    val train_data_collected: Array[(Long, (Int, Array[Double]))] = train_data.collect()

    //FIXME - are the index values needed or will the order always be in the right direction?
    val minorityData: RDD[(Long, Int, String, Array[Double])] = train_data.map(x => getDistances(x, train_data_collected)).cache() //FIXME try not caching this?
    //FIXME -return indicies per class/minority type

    val results = minorityData.toDF()
    results.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")

  }
}
  