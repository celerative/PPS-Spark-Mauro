import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object Ensemble_Test{
def main(args: Array[String]) {

  val conf = new SparkConf().setAppName("Ensemble").setMaster("local")
  val sc = new SparkContext(conf)
  val data = MLUtils.loadLibSVMFile(sc,"/home/mauro/IdeaProjects/Ensemble_Example/src/main/res/breast-cancer")

  //Divido la información en un 70% para training y 30% para test
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  //trainingData.collect().foreach(println)
  //Ejemplos de ensemble que ya vienen incluidos de base en las librerias de spark
  ensembleRandomForest(trainingData,testData)
  ensembleGradientBoostedTree(trainingData,testData)

}

  def ensembleRandomForest(train: RDD[org.apache.spark.mllib.regression.LabeledPoint], test:RDD[org.apache.spark.mllib.regression.LabeledPoint]): Unit ={
    // Entrenando modelo de RandomForest
    val numClasses = 5
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    //Clasificacion utilizando RandomForest,Majority vote
    val classifier = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluando el modelo
    val labelAndPreds = test.map { point =>
      val prediction = classifier.predict(point.features)
      (point.label, prediction)
    }

    //Imprimiendo la tasa de error del test
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification forest model:\n ${classifier.toDebugString}")

    //Regresion utilizando RandomForest, Average
    val regressor = RandomForest.trainRegressor(train, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, "variance", maxDepth, maxBins)

    // Evaluando el modelo
    val labelsAndPredictions = test.map { point =>
      val prediction = regressor.predict(point.features)
      (point.label, prediction)
    }

    //Imprimiendo la tasa de error del test
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Error cuadratico medio = $testMSE")
    println(s"Learned regression forest model:\n ${regressor.toDebugString}")
  }

  def ensembleGradientBoostedTree(train: RDD[org.apache.spark.mllib.regression.LabeledPoint], test:RDD[org.apache.spark.mllib.regression.LabeledPoint]): Unit ={
    // Entrenando modelo de GradientBoostedTrees

    //Clasificación
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 30
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5

    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val modelC = GradientBoostedTrees.train(train, boostingStrategy)

    // Evaluando el modelo
    val labelAndPreds = test.map { point =>
      val prediction = modelC.predict(point.features)
      (point.label, prediction)
    }

    //Imprimiendo la tasa de error del modelo
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Error = $testErr")
    println(s"Modelo GBT de Learned Regression:\n ${modelC.toDebugString}")

    //Regresion
    val boostingStrategy1 = BoostingStrategy.defaultParams("Regression")
    boostingStrategy1.numIterations = 30
    boostingStrategy1.treeStrategy.maxDepth = 5

    boostingStrategy1.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val modelR = GradientBoostedTrees.train(train, boostingStrategy1)

    // Evaluando modelo
    val labelsAndPredictions = test.map { point =>
      val prediction = modelR.predict(point.features)
      (point.label, prediction)
    }

    //Imprimiendo tasa de error del modelo
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Error cuadratico medio = $testMSE")
    println(s"Modelo GBT de Learned Regression:\n ${modelR.toDebugString}")

  }
}