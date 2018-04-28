package edu.vcu.sleeman

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.{DecisionTreeClassifier, KNNClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame

import scala.util.Random
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

//import edu.vcu.sleeman.MinorityClass._

object Classifier {

  def getIndexedDF(spark: SparkSession, columns: Array[String], trainData: DataFrame): DataFrame ={
    import spark.implicits._

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
    //val setPrimaryClass = udf {(label: Int) => 
    //  if(label.toInt.equals(cls)) 1 else 0
    // }

    //val filteredDF = data.withColumn("label", when(col("label") === cls, 1).otherwise(0))

    // filteredDF.show()
    //val Array(trainData, testData) = filteredDF.randomSplit(Array(0.8, 0.2)) //FIXME - startify

    //val Array(trainData0, testData0) = filteredDF.filter(x=>x(41)==1).randomSplit(Array(0.8, 0.2))
    //val Array(trainData0, testData0) = filteredDF.filter("label == 0").randomSplit(Array(0.8, 0.2))
    //val Array(trainData1, testData1) = filteredDF.filter("label == 1").randomSplit(Array(0.8, 0.2))

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))




    //val trainDataStratified = trainData0.union(trainData1)
    //val testDataStratified = testData0.union(testData1)

    /*println("*** splits ****")
    println(trainData0.count(), testData0.count())
    println(trainData1.count(), testData1.count())
    */
    //println(trainDataStratified.count(), testDataStratified.count())
    //val filteredDF2 = smote(spark, trainDataStratified)//underSample(trainData)underSample(trainDataStratified)//

    val filteredDF2 = samplingMethod match {
      case "undersample" => underSample(spark, trainData)
      case "oversample" => overSample(spark, trainData)
      case "smote" => smote(spark, trainData)
      case _ => trainData
    }

    println("Sampled Counts")
    val aggregatedCounts = filteredDF2.groupBy("label").agg(count("label")) //FIXME
    aggregatedCounts.show()

    //getCountsByClass(spark, "label", filteredDF2).show()

    println("test counts")
    getCountsByClass(spark, "label", testData).show()


    val inputCols = data.columns.filter(_ != "label")

    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    println("total sampled training size: " + filteredDF2.count())
    //val assembledTrainData = assembler.transform(filteredDF2)
    //val assembledTestData = assembler.transform(testData)
    val assembledTrainData = assembler.transform(filteredDF2)
    val assembledTestData = assembler.transform(testData)


    val classifier = new DecisionTreeClassifier().setMaxBins(80).
      setSeed(Random.nextLong()).
      setLabelCol("label").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val model = classifier.fit(assembledTrainData)
    val predictions = model.transform(assembledTestData)


    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    //setLabelCol("label").
    //setPredictionCol("prediction")


    //val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    //val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(f1)

    val predictionRDD = predictions.
      select("prediction", "label").
      as[(Double,Double)].rdd
    //val multiclassMetrics = new MulticlassMetrics(predictionRDD)

    val confusionMatrix = predictions.
      groupBy("label").
      pivot("prediction", (0 to maxLabels.toInt)).
      count().
      na.fill(0.0).
      orderBy("label")


    //predictions.show()




    // precision=TP / (TP + FP) 
    //sensitivity = TP / (TP + FN)
    //specificity = TN / (FP + TN) 
    //F-score = 2*TP /(2*TP + FP + FN) 

    /*println("********")

    confusionMatrix.show()
    val c = confusionMatrix.toDF()

    val z = c.map{
      row => row.getInt(0)
    }
    z.show()*/
    /*for(cls<-0 to maxLabels.toInt) {
      println("***", cls)
      
      val sensitivity = 0
      val specificity = 0
      
    }*/

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

    //val foo = indexedLabelNames.filter(indexedLabelNames("label").equalTo(0))
    //indexedLabelNames.filter(indexedLabelNames("label").equalTo("200"))
    //indexedLabelNames.show()

    println("\nClass " + name + "\n" + positiveResults + "\n" + negativeResults + "\n")
  }



  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)

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

    val train_index = trainRDD.zipWithIndex().map({case(x,y)=>(y,x)}).cache()
    //val trainIndexBroadcast = spark.broadcast(train_index.collect())

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
    for(l<-0 to numLabels-1) {
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
    /*
       val positiveCases = df.filter(df("label") === 1)
       val negativeCases = df.filter(df("label") === 0)

       if(positiveCases.count() < negativeCases.count()) {
         val ratio = negativeCases.count()/positiveCases.count().toFloat
         println("ratio: " + ratio)

         val results = positiveCases.sample(true, ratio)//.union(negativeCases)
         val totalResults = results.union(negativeCases)

         println("total positives: " + results.count())
         return totalResults
         }
       else {
         return df
       }*/
  }

  def underSample(spark: SparkSession, df: DataFrame): DataFrame ={
    println("~~~~~~~~~~~~~~~~~ Under Sample")
    val counts = getCountsByClass(spark, "label", df).collect().map(x=>x(1).toString().toInt).sorted
    counts.foreach(println)

    val undersampleCount = counts(counts.length/2).toInt
    var samples = Array[Row]()    //FIXME - make this more parallel

    for(cls<-0 to counts.length-1) {
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
    import org.apache.spark.sql.SparkSession
    import spark.implicits._

    val numClasses = df.select(df("label")).distinct().count().toInt
    val aggregatedCounts = df.groupBy("label").agg(count("label"))
    println("SMOTE counts")
    aggregatedCounts.show()
    println(numClasses + " " + aggregatedCounts.count())
    val smoteTo = 1000
    var samples = ArrayBuffer[Row]()    //FIXME - make this more parallel
    for(cls<-0 to numClasses-1) {
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
          //val ratio = samplesToAdd / currentCount

          val currentClassZipped = currentClass.collect().zipWithIndex

          var smoteSamples = ArrayBuffer[Row]()
          for(s<-1 to samplesToAdd.toInt) {
            def r = scala.util.Random.nextInt(currentClassZipped.length)
            val rand = Array(r,r,r,r,r)
            val sampled = currentClassZipped.filter(x=>(rand.contains(x._2))).map(x=>x._1) //FIXME - issues not taking duplicates
            //println(sampled(0)) - These are the correct schema
            //val xx = sampled.map(x=>x.toSeq).toList.transpose

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

    println("New total count: " + smoteDF.count())


    return smoteDF

    /*val positiveCases = df.filter(df("label") === 1)
    val negativeCases = df.filter(df("label") === 0)
    
    
    val positiveCollected = positiveCases.collect()
    
    //println(positiveCollected.length)
    val positiveZipped = positiveCollected.zipWithIndex
    
    
    //val foo = positiveCases.take(1000)
    //println(foo.length)
    //val foo2 = foo.map(x=>Array(x))//.foreach(println)
    //val bar = foo2.take(1)(0)(0)
    //println(bar)
    
//    val foo = positiveCases.rdd.sample(//withReplacement, fraction, seed)  
    val samplesToAdd = negativeCases.count() - positiveCases.count()
    if(samplesToAdd <= 0) {
      return df
    }
    else {
      println("positive cases: " + positiveCases.count())
      if(positiveCases.count() > 0) {
        val percent = 5/positiveCases.count().toDouble
        //println("percent: " + percent)
       
        var smoteSamples = ArrayBuffer[Row]()
        for(x<-1 to samplesToAdd.toInt) {
          //println(x)
          
          def r = scala.util.Random.nextInt(positiveZipped.length)
          val rand = Array(r,r,r,r,r)
          //rand.foreach(println)
          //println("~~~~~")
          val sampled = positiveZipped.filter(x=>(rand.contains(x._2))).map(x=>x._1) //FIXME - issues not taking duplicates
          //sampled.foreach(println)
          //println("********")
          //val values = positiveCases.rdd.sample(true, percent).take(5) //FIXME  // indexing trick to make it faster
          //print(values.toList.map(x=>x.toSeq))
          
         //val d = values.toList.map(x=>x.toSeq)
          
          
          val xxxxx = (sampled.toList.map(x=>x.toSeq.toList.map(_.toString().toDouble)))
         
          val ddd = xxxxx.toList.transpose.map(_.sum/xxxxx.length)
          //println("DDDDD " + ddd)
          
          //val averages = values.columns.map(avg(_))
          //val sample = values.agg(averages.head, averages.tail: _*).take(1)(0)
          val r2 = Row.fromSeq(ddd.toSeq)
          // println("****************************")
         //println(r2)
         // println("****************************")    
          smoteSamples += r2
        }
           
        
        import org.apache.spark.sql.types.{StructField, DoubleType, StructType} 
        println("Number of added SMOTE samples: " + smoteSamples.length)
        val rdd = spark.sparkContext.makeRDD(smoteSamples)
        val xxx = df.schema.map(x=>StructField(x.name, DoubleType, true))
        val foo3 = df.union(spark.createDataFrame(rdd, StructType(xxx)))
        
        println("Initial total count: " + df.count())
        println("Negative count: "+ negativeCases.count())
        println("Positive count: "+ positiveCases.count())
        println("New SMOTE samples: "+ smoteSamples.length)
        println("New total count: " + foo3.count())
        //println(foo3.count())
        //println("*****")
        //println(df.schema.take(1)(0))
       
  
  
  
        //val schema2 = StructField(df.schema.take(1)(0).name, DoubleType,true)   
        //println(schema2)
        println("*****")
        //rdd.take(1).foreach(println)
         return foo3
      }
      else {
        return df
      }
    }*/

  }


  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame ={
    import spark.implicits._

    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(100) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x=>(x(0).toString(),x(1).toString().toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    rdd.toDF()
  }


  def runSparKNN(df: DataFrame): Unit = {
    val knn = new KNNClassifier()
      .setTopTreeSize(df.count().toInt / 500)
      .setK(10)


    println("K= " + knn.getK)
  }



  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)


    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val input_file = args(0)
    val labelColumnName = args(1)
    val useHeader = if(args.length > 2 && args(2).equals("yes")) true else false

    val df1 = spark.read.
      option("inferSchema", true).
      option("header",useHeader).
      csv(input_file)

    val df = df1.repartition(8)


    ////////
    val stringColumns = df.schema.filter(x=>x.dataType.toString().equals("StringType")).map(x=>x.name).toArray

    val preppedData = getIndexedDF(spark, stringColumns, df)//.withColumnRenamed("_c41_idx", "label")



    //val x = preppedData.select(preppedData.columns.map(name => func(col(name))): _*)
    //x.show()
    //return
    ////////

    // labelColumnName
    val converter = new IndexToString()
      .setInputCol(labelColumnName)
      .setOutputCol("originalCategory")

    //////////// 
    val converted = converter.transform(preppedData)
    val foo2 = converted.select(labelColumnName, "originalCategory").distinct()
    foo2.sort("_c41_idx").show()
    ////////////

    //////////// 
    val preppedDataUpdated1 = preppedData.withColumnRenamed(if(stringColumns.contains(labelColumnName)) labelColumnName + "_idx" else labelColumnName, "label")

    def func(column: Column) = column.cast(DoubleType)
    val preppedDataUpdated = preppedDataUpdated1.select(preppedDataUpdated1.columns.map(name => func(col(name))): _*)


    runSparKNN(preppedDataUpdated)
    //getMinorityClassData(preppedDataUpdated)



    //preppedDataUpdated.show()
    //////////// 


    //println(trainData.count(), testData.count())
    //val foo3 = df.groupBy("_c41").agg(count("_c41")).take(100)

    //val sc = spark.sparkContext
    //val bar = foo3.map(x=>(x(0).toString(),x(1).toString().toInt)).toSeq
    //val rdd = sc.parallelize(bar)

    //rdd.toDF().show()

    getCountsByClass(spark, "_c41", df).show()
    //for(cls<-preppedDataUpdated.select("label").distinct()) {
    //  println(cls(0).toString().toDouble.toInt)
    // }

    //preppedDataUpdated.select("label").distinct().show()
    val clsArray = preppedDataUpdated.select("label").distinct().collect()
    //val methods = Array("None", "oversample", "undersample", "smote")
    val methods = Array("None")

    //for(cls<-clsArray) {
    //val rows: Array[Row] = foo2.collect
    // val res = rows.filter(x=>x(0).toString().toFloat==cls(0))
    //println("\n\n\n~~~~~~~~~~~~~~~~~~~" + res(0)(1) + "~~~~~~~~~~~~~~~~~~~")
    for(method<-methods) {
      println("*********** Method " + method + "***********")

      //println("Class: " + res(0)(1))
      runClassifier(spark, preppedDataUpdated, method)
    }
  }
}
  