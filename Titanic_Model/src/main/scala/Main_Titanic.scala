import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, IntegerType, StringType, StructField, StructType, DoubleType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier,
  LogisticRegression, LinearSVC, NaiveBayes, DecisionTreeClassificationModel, DecisionTreeClassifier,
  MultilayerPerceptronClassifier,GBTClassificationModel, GBTClassifier, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.evaluation.{ BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

object Main_Titanic {
  def main(args: Array[String]) {
    val dataScheme = new StructType()
      .add("PassengerId", IntegerType)
      .add("Survived", IntegerType)
      .add("Pclass", IntegerType)
      .add("Name", StringType)
      .add("Sex", StringType)
      .add("Age", FloatType)
      .add("SibSp", IntegerType)
      .add("Parch", IntegerType)
      .add("Ticket", StringType)
      .add("Fare", FloatType)
      .add("Cabin", StringType)
      .add("Embarked", StringType)

    val dataSchemeTest = new StructType()
      .add("PassengerId", IntegerType)
      .add("Pclass", IntegerType)
      .add("Name", StringType)
      .add("Sex", StringType)
      .add("Age", FloatType)
      .add("SibSp", IntegerType)
      .add("Parch", IntegerType)
      .add("Ticket", StringType)
      .add("Fare", FloatType)
      .add("Cabin", StringType)
      .add("Embarked", StringType)

    val spark = SparkSession.builder.master("local[*]").appName("Titanic").getOrCreate()
    val sqlContext = spark.sqlContext

    val datasetDF = sqlContext.read.schema(dataScheme).option("header", "true").csv("src/main/res/train.csv")
    datasetDF.createOrReplaceTempView("train")

    val datasetDFTest = sqlContext.read.schema(dataSchemeTest).option("header","true").csv("src/main/res/test.csv")
    datasetDFTest.createOrReplaceTempView("test")


    //Las siguientes lineas imprimen en pantalla la relacion de Pclass,Sex,SibSp y Parch con Survived para ver cuantos sobrevivieron de cada uno de los tipos de esas clases
    /*val df_pclass = datasetDF.select("Pclass","Survived").groupBy("Pclass").agg(avg("Survived")).orderBy(desc("avg(Survived)"))
    df_pclass.show()
    val df_sex = datasetDF.select("Sex","Survived").groupBy("Sex").agg(avg("Survived")).orderBy(desc("avg(Survived)"))
    df_sex.show()
    val df_sibsp = datasetDF.select("SibSp","Survived").groupBy("SibSp").agg(avg("Survived")).orderBy(desc("avg(Survived)"))
    df_sibsp.show()
    val df_parch = datasetDF.select("Parch","Survived").groupBy("Parch").agg(avg("Survived")).orderBy(desc("avg(Survived)"))
    df_parch.show()*/


    //Calculo de la media de la edad (age) para completar el dataset
    val averageAge = datasetDF.select("Age")
      .agg(avg("Age"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    //Calculo de la media de la tarifa (fare) para completar el dataset
    val averageFare = datasetDF.select("Fare")
      .agg(avg("Fare"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    val fillNAMap = Map(
      "Fare" -> averageFare,
      "Age" -> averageAge,
      "Embarked" -> "S"
    )

    val datasetDFfill = datasetDF.na.fill(fillNAMap)
    val datasetDFTestfill = datasetDFTest.na.fill(fillNAMap)

    val datasetDFdrop = datasetDFfill.drop("Ticket","Cabin","PassengerId")
    val datasetDFTestdrop = datasetDFTestfill.drop("Ticket","Cabin")

    //Funciones para poder sustraer el Titulo de cada persona
    //Mala practica
    //val titleDF = afterdropDF.withColumn("Tit",substring_index(col("Name"),".",1)).withColumn("Title",substring_index(col("Tit"),",",-1)).drop("Tit")

    //Buena practica
    //regexp_extract devuelve una ocurrencia a la izquierda con la regular expression definida en el segundo argumento
    val datasetDFTitle = datasetDFdrop.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\\.", 1)).drop("Name")
    val datasetDFTestTitle = datasetDFTestdrop.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\\.", 1)).drop("Name")

    //Se muestran los distintos tipos de titulos que hay entre las personas a bordo
    //titleDF.select("Title").distinct().show()

    //Se realiza una tabla cruzada entre Title y Sex, para ver cuantas personas de cada uno de los sexos hay para cada Title
    //titleDF.stat.crosstab("Title","Sex").show()

    val titles = Map(
      "Don" -> "Rare",
      "Countess" -> "Rare",
      "Col" -> "Rare",
      "Rev" -> "Rare",
      "Lady" -> "Rare",
      "Mme" -> "Mrs",
      "Capt" -> "Rare",
      "Dr" -> "Rare",
      "Sir" -> "Rare",
      "Jonkheer" -> "Rare",
      "Mlle" -> "Miss",
      "Major" -> "Rare",
      "Ms" -> "Miss"
    )


    val datasetDFReplace = datasetDFTitle.na.replace("Title", titles)
    val datasetDFTestReplace = datasetDFTestTitle.na.replace("Title",titles)
    //Para replace no se puede mapear de un string a un int..Investigar
    //val titleindex = title_replace.na.replace("Title",title_index)

    //Se realiza una tabla mostrando la media de sobrevivientes para cada tipo de title
    //datasetDF.select("Title", "Survived").groupBy("Title").agg(avg("Survived")).orderBy(desc("avg(Survived)")).show()

    /*
    //Indexado a traves de un mapeo de string a int
    val title_index: (String => Int) = {
      case "Mr" => 1
      case "Miss" => 2
      case "Mrs" => 3
      case "Master" => 4
      case "Rare" => 5
      case "null" => 6
    }
    val title_indexUDF = udf(title_index)

    val sex: (String => Int) = {
      case "female" => 1
      case "male" => 0
      case "null" => 2
    }
    val sexUDF = udf(sex)

    val datasetDFIndexT = datasetDFReplace.withColumn("Title",title_indexUDF(datasetDFReplace.col("Title")))
    val datasetDFIndexS = datasetDFIndexT.withColumn("Sex",sexUDF(datasetDFIndexT.col("Sex")))
    val datasetDFTestIndexT = datasetDFTestReplace.withColumn("Title",title_indexUDF(datasetDFTestReplace.col("Title")))
    val datasetDFTestIndexS = datasetDFTestIndexT.withColumn("Sex",sexUDF(datasetDFTestIndexT.col("Sex")))*/
    //datasetDF.show()
    //datasetDF.printSchema()


    val datasetDFIndexA = datasetDFReplace.withColumn("Age",when(expr("Age <= 16"),0)
      .when(expr("Age > 16 and Age <=32"),1)
      .when(expr("Age > 32 and Age <=48"),2)
      .when(expr("Age > 48 and Age <=64"),3).otherwise(4))
    //datasetDF = datasetDF.withColumn("Age", ageUDF(datasetDF.col("Age")))

    val datasetDFTestIndexA = datasetDFTestReplace.withColumn("Age",when(expr("Age <= 16"),0)
      .when(expr("Age > 16 and Age <=32"),1)
      .when(expr("Age > 32 and Age <=48"),2)
      .when(expr("Age > 48 and Age <=64"),3).otherwise(4))


    val datasetDFFamily=datasetDFIndexA.withColumn("FamilySize",col("SibSp")+col("Parch"))
    val datasetDFAlone=datasetDFFamily.withColumn("Alone",when(expr("FamilySize == 0"),1).otherwise(0))

    val datasetDFTestFamily = datasetDFTestIndexA.withColumn("FamilySize",col("SibSp")+col("Parch"))
    val datasetDFTestAlone = datasetDFTestFamily.withColumn("Alone",when(expr("FamilySize == 0"),1).otherwise(0))
    //Mostrar correlacion entre familysize y survived y con isalone y survived

    val datasetDFDrop2 = datasetDFAlone.drop("Parch","SibSp","FamilySize")
    val datasetDFAC = datasetDFDrop2.withColumn("Age*Class",col("Age")*col("Pclass"))

    val datasetDFTestDrop2 = datasetDFTestAlone.drop("Parch","SibSp","FamilySize")
    val datasetDFTestAC = datasetDFTestDrop2.withColumn("Age*Class",col("Age")*col("Pclass"))

    /*val embarked: (String => Int) = {
      case "C" => 1
      case "S" => 0
      case "Q" => 2
    }
    val embarkedUDF = udf(embarked)

    val datasetDFIndexE = datasetDFAC.withColumn("Embarked",embarkedUDF(datasetDFAC.col("Embarked")))*/
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("Embarked_indexed")
    val titleIndexer = new StringIndexer().setInputCol("Title").setOutputCol("Title_indexed")
    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("Sex_indexed")
    val dataTrain = datasetDFAC.withColumn("Fare",when(expr("Fare <= 7.91"),0)
      .when(expr("Fare > 7.91 and Fare <= 14.454"),1)
      .when(expr("Fare > 14.454 and Fare <= 31"),2)
      .otherwise(3))

    //val datasetDFTestIndexE = datasetDFTestAC.withColumn("Embarked",embarkedUDF(datasetDFTestAC.col("Embarked")))

    val dataTestt = datasetDFTestAC.withColumn("Fare",when(expr("Fare <= 7.91"),0)
      .when(expr("Fare > 7.91 and Fare <= 14.454"),1)
      .when(expr("Fare > 14.454 and Fare <= 31"),2)
      .otherwise(3))

    val pipeline = new Pipeline().setStages(Array(embarkedIndexer,titleIndexer,sexIndexer))
    val model = pipeline.fit(dataTrain)
    val trainingData= model.transform(dataTrain).drop("Sex","Title","Embarked")
   /* val trainingData = trainingData1.withColumn("Sex_indexed", trainingData1.col("Sex_indexed").cast(IntegerType))
      .withColumn("Title_indexed", trainingData1.col("Title_indexed").cast(IntegerType))
      .withColumn("Embarked_indexed", trainingData1.col("Embarked_indexed").cast(IntegerType))*/
    //trainingData.show()


    val pipelineTest = new Pipeline().setStages(Array(embarkedIndexer,titleIndexer,sexIndexer))
    val modelTest = pipelineTest.fit(dataTestt)
    val testData= modelTest.transform(dataTestt).drop("Sex","Title","Embarked","PassengerId").withColumn("Survived",lit(0))

    //Pasando del tipo double de los indexed a tipo integer
    /*val testData = testData1.withColumn("Sex_indexed", testData1.col("Sex_indexed").cast(IntegerType))
      .withColumn("Title_indexed", testData1.col("Title_indexed").cast(IntegerType))
      .withColumn("Embarked_indexed", testData1.col("Embarked_indexed").cast(IntegerType))*/
    //testData.show()

    //val Array(trainingData, testData) = dataTrain1.randomSplit(Array(0.8, 0.2))

    //Declaracion de columnas con las caracteristicas a tener en cuenta para entrenar el dataframe
    val featureCols = Array( "Pclass", "Age","Sex_indexed", "Fare", "Embarked_indexed", "Title_indexed", "Alone","Age*Class" )
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    //Declaracion de la feature a entrenar, designandola como label
    val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")

    //Realización de regresión lógica
    val lr = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

    val ovr = new OneVsRest().setClassifier(lr)

    val pipelineLr = new Pipeline().setStages(Array(assembler, labelIndexer, ovr))

    val paramGridLr = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).build()

    //val predictLr = classificationV(pipelineLr,trainingData,testData)

    //val accuracyLR = classificationAccuracyMCE(predictLr)
    val accuracyLR = classification(pipelineLr,trainingData,testData,paramGridLr)
    //Realización de Support Vector Machine
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    val pipeLsvc = new Pipeline().setStages(Array(assembler,labelIndexer,lsvc))

    val paramGridLsvc = new ParamGridBuilder().addGrid(lsvc.regParam,
      Array(0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)).build()

    //val predictLsvc = classificationV(pipeLsvc,trainingData,testData)

    //val accuracyLSVC = classificationAccuracyMCE(predictLsvc)
    val accuracyLSVC = classification(pipeLsvc,trainingData,testData,paramGridLsvc)
    // Realización de NaiveBayes
    val nb = new NaiveBayes()
    val pipeNB = new Pipeline().setStages(Array(assembler,labelIndexer,nb))

    val paramGridNB = new ParamGridBuilder().addGrid(nb.smoothing, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).build()
    //val predictNB = classificationV(pipeNB,trainingData,testData)

    //val accuracyNB = classificationAccuracyMCE(predictNB)
    val accuracyNB = classification(pipeNB,trainingData,testData,paramGridNB)
    //Realización de Decision Tree
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeDt = new Pipeline()
      .setStages(Array(labelIndexer, assembler, dt))

    val paramGridDt = new ParamGridBuilder()
      .addGrid(dt.maxBins, Array(5, 7))
      .build()

    //val predictDt = classificationV(pipeDt,trainingData,testData)

    //val accuracyDT = classificationAccuracyMCE(predictDt)
    val accuracyDT = classification(pipeDt,trainingData,testData,paramGridDt)
    //Realización de Randomn Forest
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeRf = new Pipeline().setStages(Array(labelIndexer, assembler, rf))

    val paramGridRF = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(25, 28, 31))
      .addGrid(rf.maxDepth, Array(4, 6, 8))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    //val predictRf = classificationV(pipeRf,trainingData,testData)

    //val accuracyRF = classificationAccuracyMCE(predictRf)
    val accuracyRF = classification(pipeRf,trainingData,testData,paramGridRF)
    //Realización de perceptron, Artificial neural network
    val layers = Array[Int](8, 5, 4, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val pipeP = new Pipeline().setStages(Array(labelIndexer,assembler,trainer))

    val paramGridMPC= new ParamGridBuilder().build()
    //val predictP = classificationV(pipeP,trainingData,testData)

    //val accuracyP = classificationAccuracyMCE(predictP)
    val accuracyP = classification(pipeP,trainingData,testData,paramGridMPC)
    //Realizacion de Gradient Boost Tree
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setFeatureSubsetStrategy("auto")

    val pipeGbt = new Pipeline().setStages(Array(labelIndexer, assembler, gbt))

    val paramGridGbt= new ParamGridBuilder()
      .addGrid(gbt.maxDepth,Array(2, 5))
      .addGrid(gbt.maxIter,Array(10, 100)).build()
    //val predictGbt = classificationV(pipeGbt,trainingData,testData)

    //val accuracyGbt = classificationAccuracyMCE(predictGbt)
    val accuracyGbt = classification(pipeGbt,trainingData,testData,paramGridGbt)
    val accSchema = StructType(List(
      StructField("Model", StringType, nullable = true),
      StructField("Score", DoubleType, nullable = false)
    )
    )
    val accrdd = sqlContext.sparkContext.parallelize(List(
      Row("Logistic Regression", accuracyLR ),
      Row("Support Vector Machine" , accuracyLSVC),
      Row("Naive Bayes" , accuracyNB),
      Row("Decision Tree" , accuracyDT),
      Row("Random Forest" , accuracyRF),
      Row("Gradient Boost Tree" , accuracyGbt),
      Row("Perceptron", accuracyP)
    ))

    val accuracyDf = sqlContext.createDataFrame(accrdd, accSchema)

    accuracyDf.orderBy(desc("Score")).show()
  }

  def classification (a: Pipeline, trainD: DataFrame, testD: DataFrame, p:Array[ParamMap]): Double ={
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val cv = new CrossValidator()
      .setEstimator(a)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(p)
      .setNumFolds(3)

    val model = cv.fit(trainD)

    val predict = model.transform(testD)

    predict.show()

    //return predict
    val accuracy = evaluator.evaluate(predict)

    println(s"Acurracy from previous dataframe = $accuracy")
    println(s"Test Error = ${(1.0 - accuracy)}")
    return accuracy
  }

  def classificationV(a: Pipeline, trainD: DataFrame, testD: DataFrame): DataFrame={
    val model = a.fit(trainD)
    val predict = model.transform(testD)

    predict.show()

    return predict
  }
  def classificationAccuracyMCE (dat: DataFrame): Double ={
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(dat)

    println(s"Acurracy from previous dataframe = $accuracy")
    println(s"Test Error = ${(1.0 - accuracy)}")

    return accuracy
  }

  def classificationAccuracyBCE (dat: DataFrame): Double ={
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(dat)

    println(s"Acurracy from previous dataframe = $accuracy")
    println(s"Test Error = ${(1.0 - accuracy)}")

    return accuracy
  }
}
