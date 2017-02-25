import org.apache.spark.sql.SparkSession

object clean_test_set {

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

    val query =
      """
        SELECT
           PassengerId,
           COALESCE(Pclass, 0) AS class,
          DENSE_RANK() OVER (ORDER BY COALESCE(Sex, 'Unknown')) as sex,
          DENSE_RANK() OVER
             (ORDER BY regexp_extract(COALESCE(Name, ' Unknown.'), ' ([A-Za-z]+)\.'))
           AS title,
           DENSE_RANK() OVER (ORDER BY COALESCE(CAST(Age / 20 AS int), 0)) AS age,
           DENSE_RANK() OVER
             (ORDER BY
               CAST(
                 (COALESCE(SibSp, 0) + COALESCE(Parch, 0)) / 3
                 AS int
               )
             ) AS family_size,
           DENSE_RANK() OVER (ORDER BY trim(regexp_replace(Cabin, '([0-9]|\/|\\.|\s)+', ''))) as cabin,
           DENSE_RANK() OVER (ORDER BY split(Race, ', ')[1]) AS race2
        FROM test
      """

    spark.sql(query).show(10)
    spark.sql(query).write.format("csv").save("data/test_clean")

  }

}
