import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{BaggingClassificationModel, BaggingClassifier, DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassificationModel, GBTClassifier, LinearSVC, LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, OneVsRest, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Fifa_18 {
  def main(args: Array[String]){
    val dataScheme = new StructType()
    .add("Name",StringType)
    .add("Age",IntegerType)
    .add("Nationality",StringType)
    .add("Overall",IntegerType)
    .add("Potential",IntegerType)
    .add("Club",StringType)
    .add("Value",IntegerType)
    .add("Acceleration",IntegerType)
    .add("Aggression",IntegerType)
    .add("Agility",IntegerType)
    .add("Balance",IntegerType)
    .add("Ball control",IntegerType)
    .add("Composure",IntegerType)
    .add("Crossing",IntegerType)
    .add("Curve",IntegerType)
    .add("Dribbling",IntegerType)
    .add("Finishing",IntegerType)
    .add("Free kick accuracy",IntegerType)
    .add("GK diving",IntegerType)
    .add("GK handling",IntegerType)
    .add("GK kicking",IntegerType)
    .add("GK positioning",IntegerType)
    .add("GK reflexes",IntegerType)
    .add("Heading accuracy",IntegerType)
    .add("Interceptions",IntegerType)
    .add("Jumping",IntegerType)
    .add("Long passing",IntegerType)
    .add("Long shots",IntegerType)
    .add("Marking",IntegerType)
    .add("Penalties",IntegerType)
    .add("Positioning",IntegerType)
    .add("Reactions",IntegerType)
    .add("Short passing",IntegerType)
    .add("Shot power",IntegerType)
    .add("Sliding tackle",IntegerType)
    .add("Sprint speed",IntegerType)
    .add("Stamina",IntegerType)
    .add("Standing tackle",IntegerType)
    .add("Strength",IntegerType)
    .add("Vision",IntegerType)
    .add("Volleys",IntegerType)

    val spark = SparkSession.builder.master("local[*]").appName("Ensemble").getOrCreate()
    val sqlContext = spark.sqlContext

    val datasetDF = sqlContext.read.schema(dataScheme).option("header", "true").csv("src/main/res/fifa.csv")
    datasetDF.createOrReplaceTempView("fifa")

    //datasetDF.describe().show()

    val meanOverall = datasetDF.agg(mean(datasetDF("Overall"))).first.getDouble(0)
    val fixedDf = datasetDF.na.fill(meanOverall, Array("Overall"))

    //fixedDf.describe().show()
    //fixedDf.select("Overall","Club").groupBy("Club").agg(avg("Overall")).orderBy(desc("avg(Overall)")).show()
    //fixedDf.select("Value","Club").groupBy("Club").agg(avg("Value")).orderBy(desc("avg(Value)")).show()

    val finalDF1 = fixedDf.drop("Name","Club","Nationality").withColumn("weight",lit(0.5))

    finalDF1.show()

    val finalDF = finalDF1.withColumn("Overall",when(expr("Overall <= 74"),0)
      .when(expr("Overall > 74 and Overall <= 83"),1)
      .otherwise(2))

    val featureCols = Array( "Age", "Potential", "Value", "Acceleration", "Aggression", "Agility",
      "Balance", "Ball control", "Composure", "Crossing", "Curve", "Dribbling", "Finishing", "Free kick accuracy",
      "GK diving", "GK handling", "GK kicking", "GK positioning", "GK reflexes", "Heading accuracy", "Interceptions",
      "Jumping", "Long passing", "Long shots", "Marking", "Penalties", "Positioning", "Reactions", "Short passing",
      "Shot power", "Sliding tackle", "Sprint speed", "Stamina", "Standing tackle", "Strength", "Vision", "Volleys", "weight")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val stringIndexer = new StringIndexer()
      .setInputCol("Overall")
      .setOutputCol("label")

    val data = stringIndexer.fit(finalDF).transform(vectorAssembler.transform(finalDF))

    val Array(train, test) = data.randomSplit(Array(0.7, 0.3))

    val baseClassifier = new DecisionTreeClassifier()
      .setMaxDepth(20).setMaxBins(30)

    //Bagging
    val baggingClassifier = new BaggingClassifier()
      .setBaseLearner(baseClassifier)
      .setMaxIter(10)
      .setParallelism(4)

    val model = baggingClassifier.fit(train)

    val predicted = model.transform(test)
    predicted.show()

    val re = new MulticlassClassificationEvaluator()

    println(re.evaluate(predicted))

    val paramGrid = new ParamGridBuilder()
      .addGrid(baggingClassifier.sampleRatioFeatures, Array(0.3,0.7,1))
      .addGrid(baggingClassifier.replacementFeatures, Array(x = false))
      .addGrid(baggingClassifier.replacement, Array(true,false))
      .addGrid(baggingClassifier.sampleRatio, Array(0.3, 0.7, 1))
      .build()

    val cv = new CrossValidator()
      .setEstimator(baggingClassifier)
      .setEvaluator(re.setLabelCol(baggingClassifier.getLabelCol)
        .setPredictionCol(baggingClassifier.getPredictionCol))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(4)

    val cvModel = cv.fit(data)

    println(cvModel.avgMetrics.mkString(","))
    print(cvModel.bestModel.asInstanceOf[BaggingClassificationModel].getReplacement + ",")
    print(cvModel.bestModel.asInstanceOf[BaggingClassificationModel].getSampleRatio + ",")
    print(cvModel.bestModel.asInstanceOf[BaggingClassificationModel].getReplacementFeatures + ",")
    println(cvModel.bestModel.asInstanceOf[BaggingClassificationModel].getSampleRatioFeatures)
    println(cvModel.avgMetrics.max)
  }
}
