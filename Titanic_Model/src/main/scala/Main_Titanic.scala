//imṕortar librerias para machine learning
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, LinearSVC, NaiveBayes}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
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


    /*datasetDF.show()
    datasetDF.describe().show()*/

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

    val datasetDFfill = datasetDF.na.fill(fillNAMap)//.withColumn("Embarked", embarkedUDF(datasetDF.col("Embarked")))
    //val Array(trainingData, testData) = filledDF.randomSplit(Array(0.7, 0.3))
    val datasetDFTestfill = datasetDFTest.na.fill(fillNAMap)

    val datasetDFdrop = datasetDFfill.drop("Ticket","Cabin","PassengerId")
    val datasetDFTestdrop = datasetDFTestfill.drop("Ticket","Cabin")
    //afterdropDF.show()

    //Funciones para poder sustraer el Titulo de cada persona
    //Mala practica
    //val titleDF = afterdropDF.withColumn("Tit",substring_index(col("Name"),".",1)).withColumn("Title",substring_index(col("Tit"),",",-1)).drop("Tit")

    //Buena practica
    //regexp_extract devuelve una ocurrencia a la izquierda con la regular expression definida en el segundo argumento
    val datasetDFTitle = datasetDFdrop.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\\.", 1)).drop("Name")
    val datasetDFTestTitle = datasetDFTestdrop.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\\.", 1)).drop("Name")
    //titleDF.show()
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
    val datasetDFTestIndexS = datasetDFTestIndexT.withColumn("Sex",sexUDF(datasetDFTestIndexT.col("Sex")))
    //datasetDF.show()
    //datasetDF.printSchema()

   /* val age: (Float => Int) = (x:Float) => {
      case (x <=16) => 0
      case ((x>16)and(x<=32)) => 1
      case ((x>32)and(x<=48)) => 2
      case ((x>48)and(x<=64)) => 3
      case (x>64) => 4
    }
    val ageUDF = udf(age_)*/

    val datasetDFIndexA = datasetDFIndexS.withColumn("Age",when(expr("Age <= 16"),0)
      .when(expr("Age > 16 and Age <=32"),1)
      .when(expr("Age > 32 and Age <=48"),2)
      .when(expr("Age > 48 and Age <=64"),3).otherwise(4))
    //datasetDF = datasetDF.withColumn("Age", ageUDF(datasetDF.col("Age")))

    val datasetDFTestIndexA = datasetDFTestIndexS.withColumn("Age",when(expr("Age <= 16"),0)
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

    val embarked: (String => Int) = {
      case "C" => 1
      case "S" => 0
      case "Q" => 2
    }
    val embarkedUDF = udf(embarked)

    val datasetDFIndexE = datasetDFAC.withColumn("Embarked",embarkedUDF(datasetDFAC.col("Embarked")))
    val dataTrain = datasetDFIndexE.withColumn("Fare",when(expr("Fare <= 7.91"),0)
      .when(expr("Fare > 7.91 and Fare <= 14.454"),1)
      .when(expr("Fare > 14.454 and Fare <= 31"),2)
      .otherwise(3))

    val datasetDFTestIndexE = datasetDFTestAC.withColumn("Embarked",embarkedUDF(datasetDFTestAC.col("Embarked")))
    val dataTestt = datasetDFTestIndexE.withColumn("Fare",when(expr("Fare <= 7.91"),0)
      .when(expr("Fare > 7.91 and Fare <= 14.454"),1)
      .when(expr("Fare > 14.454 and Fare <= 31"),2)
      .otherwise(3))

    //dataTrain.show()
    //dataTest.show()
    val dataTest = dataTestt.drop("PassengerId")
    //val dataTrain = dataTrainT.drop("Survived")

    val Array(trainingData, testData) = dataTrain.randomSplit(Array(0.8, 0.2))
    //Declaracion de columnas con las caracteristicas a tener en cuenta para entrenar el dataframe
    val featureCols = Array( "Pclass","Sex", "Age", "Fare", "Embarked", "Title", "Alone","Age*Class" )
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    //Declaracion de la feature a entrenar, designandola como label
    val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")
    //val labelIndexerT= new StringIndexer().setInputCol("Sex").setOutputCol("label").fit(dataTest)

    //Realización de regresión lógica
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, lr))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.show()

    val evaluatorLR = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracyLR = evaluatorLR.evaluate(predictions)

    println(s"Logistic Reggresion accuracy = $accuracyLR")
    //Realización de Support Vector Machine
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
    val pipeLsvc = pipeline.setStages(Array(assembler,labelIndexer,lsvc))

    val lsvcModel = pipeLsvc.fit(trainingData)

    val predict = lsvcModel.transform(testData)

    predict.show()

    val evaluatorLSVC = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracyLSVC = evaluatorLSVC.evaluate(predict)

    println(s"Linear SVC = $accuracyLSVC")
    // Realización de NaiveBayes
    val nb = new NaiveBayes()
    val pipeNB = pipeline.setStages(Array(assembler,labelIndexer,nb))

    val nBModel = pipeNB.fit(trainingData)

    val predictNB = nBModel.transform(testData)

    predictNB.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictNB)
    println(s"Naive Bayes accuracy = $accuracy")
  }
}
