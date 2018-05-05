package edu.vcu.sleeman

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
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


//import edu.vcu.sleeman.MinorityClass._

//FIXME - turn classes back to Ints instead of Doubles

object Classifier {

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)


  def getIndexedDF(spark: SparkSession, columns: Array[String], trainData: DataFrame): DataFrame ={
    val keepCols = trainData.columns.map(status => if (columns.contains(status) && !status.equals("_c41")) None else status)
    val indexers = columns.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_idx")
    }
    val pipeline = new Pipeline()
      .setStages(indexers)

    val newDF = pipeline.fit(trainData).transform(trainData)
    val filteredDF = newDF.select(newDF.columns .filter(colName => !columns.contains(colName)) .map(colName => new Column(colName)): _*)
    filteredDF
  }


  def runClassifier(spark: SparkSession, data: DataFrame, samplingMethod: String) {
    import spark.implicits._

    val maxLabels = data.select("label").distinct().count()

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2)) //FIXME - does this need to be stratified?
    println("train data size: " + trainData.count())
    // Sample data
    val filteredDF2 = samplingMethod match {
      case "undersample" => underSample(spark, trainData)
      case "oversample" => overSample(spark, trainData)
      case "smote" => smote(spark, trainData)
      case _ => trainData
    }

    val maxLabel: Int = testData.select("label").distinct().collect().map(x=>x.toSeq.last.toString().toDouble.toInt).max
    //println(maxLabel)
    //println("~~~~~~~~~~~~~~")
    println("train data size after: " + filteredDF2.count())
    println("test data size: " + testData.count())


    println("Sampled Counts")
    val aggregatedCounts = filteredDF2.groupBy("label").agg(count("label")) //FIXME
    aggregatedCounts.show()

    println("test counts")
    getCountsByClass(spark, "label", testData).show()

    val inputCols = data.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    println("total sampled training size: " + filteredDF2.count())
    val assembledTrainData = assembler.transform(filteredDF2)
    val assembledTestData = assembler.transform(testData)

    /*val classifier = new DecisionTreeClassifier().setMaxBins(80).
      setSeed(Random.nextLong()).
      setLabelCol("label").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")
*/
    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(Random.nextLong()).
      setLabelCol("label").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")


    val model = classifier.fit(assembledTrainData)
    val predictions = model.transform(assembledTestData)


    val testLabels = testData.select("label").distinct().map(_.getAs[Double]("label")).map(x=>x.toInt).collect().sorted
    //println("** here **")
    ///testLabels.foreach(println)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")

    //val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    //println(f1)

    val predictionRDD = predictions.
      select("prediction", "label").
      as[(Double,Double)].rdd

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
    //val xxx: Array[Seq[Any]] = confusionMatrix.collect.map(_.toSeq)

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x=>x.tail.map(y=>y.toDouble.toInt).sum).sum

    val classMaps = testLabels.zipWithIndex.map(x=>(x._2, x._1))
    classMaps.foreach(println)
    //for(clsIndex<-0 to testLabels.length - 1) {

    var sensitiviySum = 0.0
    var sensitiviyCount = 0
    //FIXME - could be made parallel w/udf
    for(clsIndex<-0 to maxLabel) {
      //print(clsIndex)
      //println("\t" + (if(testLabels.contains(clsIndex)) 1 else 0))

      val colSum = rows.map(x=>x(clsIndex+1).toInt).sum
      val rowValueSum = if(classMaps.map(x=>x._2).contains(clsIndex)) rows.filter(x=>x(0).toDouble.toInt==clsIndex)(0).tail.map(x=>x.toDouble.toInt).sum else 0
      val tp = if(classMaps.map(x=>x._2).contains(clsIndex)) rows.filter(x=>x(0).toDouble.toInt==clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn = colSum - tp
      val fp = rowValueSum - tp
      val tn = totalCount - tp - fp - fn

      println("tp: " + tp + " fp: " + fp + " tn: " + tn +  " fn: " + fn)
      val sensitivity = tp / (tp + fn).toFloat
      //println(sensitivity)
      if(tp + fn > 0) {
        sensitiviySum += sensitivity
        sensitiviyCount += 1
      }

    }
    println(sensitiviyCount + " " + sensitiviySum)
    println("AvAcc: " + sensitiviySum/sensitiviyCount)
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

  def setMinorityStatus(cls: Int, sample: (Int, Array[Int])): (Int, String) ={
    //positive case
    if(cls == sample._1) {
      val matchingCount = sample._2.filter(x=>x==cls).length
      (1, getMinorityClassLabel(matchingCount))
    }
    //negative case
    else {
      val matchingCount = sample._2.filter(x=>x!=cls).length
      (0, getMinorityClassLabel(matchingCount))
    }
  }

  def getMinorityClassResults(cls: Int, name: String, data: Array[NearestClassResult]) {
    val minorityStatus = data.map(x=>setMinorityStatus(cls, x))

    val positiveSamples = minorityStatus.filter(x=>x._1==1).map(x=>x._2)
    val negativeSamples = minorityStatus.filter(x=>x._1!=1).map(x=>x._2)

    val positiveSafeCount = positiveSamples.filter(x=>(x=="safe")).length
    val positiveBorderlineCount = positiveSamples.filter(x=>(x=="borderline")).length
    val positiveRareCount = positiveSamples.filter(x=>(x=="rare")).length
    val positiveOutlierCount = positiveSamples.filter(x=>(x=="outlier")).length
    val positiveResults = "Positive: " + positiveSamples.length + "\nsafe: "+  positiveSafeCount  + " borderline: " + positiveBorderlineCount  + " rare: " + positiveRareCount + " outlier: " +  positiveOutlierCount

    val negativeSafeCount = negativeSamples.filter(x=>(x=="safe")).length
    val negativeBorderlineCount = negativeSamples.filter(x=>(x=="borderline")).length
    val negativeRareCount = negativeSamples.filter(x=>(x=="rare")).length
    val negativeOutlierCount = negativeSamples.filter(x=>(x=="outlier")).length
    val negativeResults = "Negative: " + negativeSamples.length + "\nsafe: " +  negativeSafeCount  + " borderline: " + negativeBorderlineCount  + " rare: " + negativeRareCount + " outlier: " +  negativeOutlierCount

    println("\nClass " + name + "\n" + positiveResults + "\n" + negativeResults + "\n")
  }

  def getDistances2(current:Element, train:Array[Element]): NearestClassResultIndex ={
    val result = train.map(x => getDistanceValue(x, current)).sortBy(x=>x._1).take(5)// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    val cls = current._2._1
    val sum = result.filter(x=>(x._2==cls)).length
    return (current._1, current._2._1, result.map(x=>x._2))
  }

  def getDistanceValue(train:Element, test:Element) : DistanceResult={
    if(train._1 == test._1) {
      return (Float.MaxValue, train._2._1)
    }
    else {
      var zipped = test._2._2.zip(train._2._2)
      var result = zipped.map({case(x,y)=>(x-y)*(x-y)})
      return ((result.sum), train._2._1) //removed sqrt
    }
  }

  def mapRow(currentRow: Array[Any])= {
    val reverseRow = currentRow.reverse
    val cls = reverseRow.head.toString().toFloat.toInt
    val features = reverseRow.tail.map(_.toString().toFloat)
    (cls, features)
  }

  def calculateMinorityClasses(spark: SparkSession, trainData: DataFrame) {
    trainData.show()

    val trainRDD = trainData.rdd.map(_.toSeq.toArray).map(x=>mapRow(x))
    trainRDD.count()

    //FIXME - is this better with broadcasting?
    val train_index = trainRDD.zipWithIndex().map({case(x,y)=>(y,x)}).cache()

    println("******************** Class Stats *****************************")
    val countOfClasses = trainRDD.map((_, 1L)).reduceByKey(_ + _).map{ case ((k, v), cnt) => (k, (v, cnt)) }.groupByKey
    //val maxCount = countOfClasses.map(x=>x._2.count(x=>true)).collect().max //For resanmpling
    val countResults = countOfClasses.map(x=>(x._1, x._2.count(x=>true)))
    countResults.sortBy(x => x._1, true).collect().foreach(println);

    val classes = countResults.sortBy(x => x._1).map(x=>x._1)
    classes.foreach(println)

    println("**************************************************************")

    println("***************** Minority Class Over/Under Resample ****************************")
    val t0 = System.nanoTime()

    val train_data_collected = train_index.collect()
    val tX = System.nanoTime()
    println("Time 1: " + (tX - t0)/1.0e9 + "s")

    val minorityData = train_index.map(x=>getDistances2(x, train_data_collected)).cache()


    val minorityDataCollected = minorityData.collect()
    val indexedLabelNames = getIndexedLabelNames(trainData)
    val rows: Array[Row] = indexedLabelNames.collect

    for(cls<-classes) {
      val res = rows.filter(x=>x(0)==cls)
      println()
      getMinorityClassResults(cls, res(0)(1).toString(), minorityDataCollected.map(x=>(x._2, x._3)))
    }
  }

  def convertIndexedToName(cls: Int, indexedLabelNames: DataFrame):String ={
    val rows: Array[Row] = indexedLabelNames.collect
    val res = rows.filter(x=>x(0)==cls)
    return res(0)(1).toString()
  }

  def getIndexedLabelNames(df: DataFrame): DataFrame ={
    val converter = new IndexToString()
      .setInputCol("label")
      .setOutputCol("originalCategory")
    val converted = converter.transform(df)
    converted.select("label", "originalCategory").distinct()
  }

  def overSample(spark: SparkSession, df: DataFrame): DataFrame ={
    val numLabels = df.select("label").distinct().count().toInt
    println("num labels: " + numLabels)
    val labelCounts = df.groupBy("label").agg(count("label")).take(numLabels)
    val maxLabelCount = labelCounts.map(x=>x(1).toString().toInt).reduceLeft(_ max _)

    var samples = Array[Row]()    //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    for(l<-0 to numLabels) {
      //println(l)
      val currentCase = df.filter(df("label") === l)
      val samplesToAdd = maxLabelCount - currentCase.count()
      //println("size requested: " + samplesToAdd)
      if(currentCase.count() == 0) {
        val currentSamples = currentCase.sample(true,(maxLabelCount)).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples

      }
      else {
        val currentSamples = currentCase.sample(true, (maxLabelCount - currentCase.count())/currentCase.count().toDouble).collect()
        println("samples created: " + currentSamples.length)
        samples = samples ++ currentSamples

      }
      //FIXME - make this faster
      //samples ++ currentCase.sample(true, (maxLabelCount - currentCase.count()/currentCase.count().toDouble)).collect()
      //val totalResults = df.union(currentCase.sample(true, (maxLabelCount - currentCase.count()/currentCase.count().toDouble)))
      //println(totalResults.count())
    }
    println("new count: " + samples.length)
    import spark.implicits._

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)

    return df.union(x)
  }

  def underSample(spark: SparkSession, df: DataFrame): DataFrame ={
    println("~~~~~~~~~~~~~~~~~ Under Sample")
    val counts = getCountsByClass(spark, "label", df).collect().map(x=>x(1).toString().toInt).sorted
    counts.foreach(println)

    val undersampleCount = counts(counts.length/2).toInt
    var samples = Array[Row]()    //FIXME - make this more parallel

    for(cls<-0 to counts.length) {
      val currentClass = df.filter(df("label") === cls)
      val underSampleRatio = undersampleCount / currentClass.count().toDouble
      if(underSampleRatio < 1.0) {
        val currentSamples = currentClass.sample(false, underSampleRatio).collect()
        samples = samples ++ currentSamples
      }
      else {
        samples = samples ++ currentClass.collect()
      }
    }
    import spark.implicits._

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    return x
  }


  def smote(spark: SparkSession, df: DataFrame): DataFrame ={
    val numClasses = df.select(df("label")).distinct().count().toInt
    val aggregatedCounts = df.groupBy("label").agg(count("label"))
    println("SMOTE counts")
    aggregatedCounts.show()
    println(numClasses + " " + aggregatedCounts.count())
    val maxClassCount = aggregatedCounts.select("count(label)").collect().toSeq.map(x=>x(0).toString.toInt).max
    println(maxClassCount)
    val smoteTo = maxClassCount/2
    var samples = ArrayBuffer[Row]()    //FIXME - make this more parallel
    for(cls<-0 to numClasses) {
      print("cls: " + cls + " " )
      val cnt = aggregatedCounts.filter(aggregatedCounts("label")===cls.toDouble).collect()
      print(cnt.length)
      if(cnt.length == 1) { // make sure this cls exist in training set

        println(cnt(0)(0), cnt(0)(1))
        val currentCount = cnt(0)(1).toString().toInt

        val currentClass = df.filter(df("label")===cls)
        if(currentCount < smoteTo) {
          val samplesToAdd = smoteTo - currentCount
          println(cls + " adding " + samplesToAdd)

          val currentClassZipped = currentClass.collect().zipWithIndex

          var smoteSamples = ArrayBuffer[Row]()
          for(s<-1 to samplesToAdd.toInt) {
            def r = scala.util.Random.nextInt(currentClassZipped.length)
            val rand = Array(r,r,r,r,r)
            val sampled = currentClassZipped.filter(x=>(rand.contains(x._2))).map(x=>x._1) //FIXME - issues not taking duplicates

            val xxxxx = (sampled.toList.map(x=>x.toSeq.toList.map(_.toString().toDouble)))
            val ddd = xxxxx.toList.transpose.map(_.sum/xxxxx.length)
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


    val xxx = df.schema.map(x=>StructField(x.name, DoubleType, true))
    val smoteDF = spark.createDataFrame(rdd, StructType(xxx))

   val finalDF = underSample(spark, smoteDF)

    println("New total count: " + smoteDF.count())
    println("Final total count: " + finalDF.count())


    return finalDF
  }


  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame ={
    import spark.implicits._

    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(100) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x=>(x(0).toString(),x(1).toString().toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    rdd.toDF()
  }

  import edu.vcu.sleeman.MinorityType.getMinorityTypeStatus
  def runSparKNN(df: DataFrame): Unit = {
    getMinorityTypeStatus(df)
  }

    def runSparKNN2(df: DataFrame): Unit = {

    val path =  "/home/ford/working/spark-knn/data/mnist/mnist1k"

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    //read in raw label and features
    val rawDataset = MLUtils.loadLibSVMFile(sc, path)
      .zipWithIndex()
      .sortBy(_._2, numPartitions = 8)
      .keys
      .toDF()

    rawDataset.show()

    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val dataset =  MLUtils.convertVectorColumnsToML(rawDataset)
      .cache()
    dataset.count() //force persist

    val knn = new KNNClassifier()
      .setTopTreeSize(8 * 10)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setK(3)

    val knnModel = knn.fit(dataset)


    println(knnModel.getDistanceCol)
    val results = knnModel.transform(dataset)

    results.printSchema()

    results.show()

    /*val knn = new KNNClassifier().setTopTreeSize(df.count().toInt / 500)
      .setFeaturesCol("features")
      .setPredictionCol("predictions")

      .setK(5)


    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val dataset =  MLUtils.convertVectorColumnsToML(df)
      .cache()
    dataset.count() //force persist
    dataset.show()

    val knnModel: KNNClassificationModel = knn.fit(dataset)
    println(knnModel.numClasses)

    val predicted: DataFrame = knnModel.transform(dataset)
    println(predicted.schema)
*/
    //println("K= " + knn.getK)
  }

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)




    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
/*
    val path =  "/home/ford/working/spark-knn/data/mnist/mnist"
    val rawDataset = MLUtils.loadLibSVMFile(spark.sparkContext, path)
      .zipWithIndex()
      //.filter(_._2 < ns.max)
      .sortBy(_._2, numPartitions = 8)
      .keys
      .toDF()

    rawDataset.show()
    return
*/

    val input_file = args(0)
    val labelColumnName = args(1)
    val useHeader = if(args.length > 2 && args(2).equals("yes")) true else false

    val df1 = spark.read.
      option("inferSchema", true).
      option("header",useHeader).
      csv(input_file)

    df1.show()

    val df = df1.repartition(8)

    /*///////
    /val stringColumns = df.schema.filter(x=>x.dataType.toString().equals("StringType")).map(x=>x.name).toArray

    val preppedData = getIndexedDF(spark, stringColumns, df)//.withColumnRenamed("_c41_idx", "label")

    // labelColumnName
    val converter = new IndexToString()
      .setInputCol(labelColumnName)
      .setOutputCol("originalCategory")

    //////////// 
    val converted = converter.transform(preppedData)
    //val foo2 = converted.select(labelColumnName, "originalCategory").distinct()
    //foo2.sort("_c41_idx").show()
    ////////////

    //////////// 
    val preppedDataUpdated1 = preppedData.withColumnRenamed(if(stringColumns.contains(labelColumnName)) labelColumnName + "_idx" else labelColumnName, "label")
`*/
    val preppedDataUpdated1 = df.withColumnRenamed(labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)
    val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)

    //preppedDataUpdated1.show()
    preppedDataUpdated.show()

    //val rawDataset = MLUtils.loadLibSVMFile(spark.sparkContext, "/home/ford/working/spark-knn/data/mnist/mnist")
     // .toDF()
    //rawDataset.show()

    /*val labeled = preppedDataUpdated1.map(row => LabeledPoint(
      row.getAs[Int]("label"),
      row.getAs[org.apache.spark.ml.linalg.Vector]("features")
    )).toDF
*/
    val inputCols = preppedDataUpdated.columns.filter(_ != "label")
    //inputCols.foreach(println)

    val assembler = new VectorAssembler().
      setInputCols(Array("Elevation", "Aspect")).
      setOutputCol("features")

    val columnNames = Seq("label", "features")

    val featureVector: DataFrame = assembler.transform(preppedDataUpdated)
    val result = featureVector.select(columnNames.head, columnNames.tail: _*)
    featureVector.show()
   runSparKNN(preppedDataUpdated)
  //  labeled.show()
    //println(labeled.count())

    //runSparKNN(preppedDataUpdated)
    //getMinorityClassData(preppedDataUpdated)

    //getCountsByClass(spark, "_c41", df).show()
    /*getCountsByClass(spark, "label", preppedDataUpdated1).show()
    val clsArray = preppedDataUpdated.select("label").distinct().collect()
    val methods = Array("None", "oversample", "undersample", "smote")
    //val methods = Array("smote")

    for(method<-methods) {
      println("*********** Method " + method + "***********")
      runClassifier(spark, preppedDataUpdated, method)
    }*/
  }
}
  