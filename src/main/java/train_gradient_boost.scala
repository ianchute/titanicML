import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, PCA, StringIndexer, VectorIndexer}

object train_gradient_boost {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Titanic ML")
      .master("local")
      .getOrCreate()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/train_libsvm")

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed=1234L)
//    val trainingData = data

    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("predicted")
      .setMaxIter(10)

    // Train model. This also runs the indexers.
    val model = gbt.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predicted", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predicted")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + (accuracy * 100) + "%")

//    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
//    println("Learned classification GBT model:\n" + gbtModel.toDebugString)

//    val unknowns = spark.read.format("libsvm").load("data/test_libsvm")
//
//    // Make predictions.
//    val unknownPredictions = model.transform(unknowns)
//
//    unknownPredictions.select("label", "predictedLabel").createTempView("predictions")
//
//    val query =
//      """
//        SELECT CAST(label AS int) AS PassengerId, CAST(predictedLabel AS int) as Survived
//        FROM predictions
//      """
//
//    spark.sql(query).write.format("csv").option("header", true).save("data/predictions.csv")

  }

}
