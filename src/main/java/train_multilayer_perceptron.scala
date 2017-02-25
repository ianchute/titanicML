import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types.IntegerType

/**
  * Created by user on 2/25/17.
  */
object train_multilayer_perceptron {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Titanic ML")
      .master("local")
      .getOrCreate()

    val data = spark.read.format("libsvm")
//        .option("header", true)
//        .option("inferSchema", true)
        .load("data/train_libsvm")

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](8, 32, 2)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setSeed(1234L)
      .setMaxIter(50)
      .setFeaturesCol("features")

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)

    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

  }

  }
