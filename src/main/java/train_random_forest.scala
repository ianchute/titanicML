import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._

object train_random_forest {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Titanic ML")
      .master("local")
      .getOrCreate()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/train_libsvm")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
//    val featureIndexer = new VectorIndexer()
//      .setInputCol("features")
//      .setOutputCol("indexedFeatures")
//      .setMaxCategories(32)
//      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed=1234L)
//    val trainingData = data

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setNumTrees(350)
      .setSeed(1234L)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + (accuracy * 100) + "%")

//    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
//    println("Learned classification forest model:\n" + rfModel.toDebugString)

    val unknowns = spark.read.format("libsvm").load("data/test_libsvm")

    // Make predictions.
    val unknownPredictions = model.transform(unknowns)

    unknownPredictions.select("label", "predictedLabel").createTempView("predictions")

    val query =
      """
        SELECT CAST(label AS int) AS PassengerId, CAST(predictedLabel AS int) as Survived
        FROM predictions
      """

    spark.sql(query).write.format("csv").option("header", true).save("data/predictions")

  }

}
