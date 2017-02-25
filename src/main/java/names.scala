import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2/26/17.
  */
object names {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Titanic ML")
      .master("local")
      .getOrCreate()

    val data = spark.read
      .format("csv")
      .option("header", true)
      .load("data/test.csv")

    data.createTempView("test")

    spark.sql(
      """
        SELECT
          trim(
            concat(
             regexp_replace(
               split(Name, ', ')[1],
               '(.+\\.)|(.+\\()|(\\)|(\\s[A-Z]\\b)|(\\"))',
               ''
             ),
             ' ',
             split(
              regexp_replace(Name, '.+\\(', ','),
            ',')[0])
          )
          as full_name
        FROM test
      """
    ).write.format("csv").save("data/names")

  }

}
