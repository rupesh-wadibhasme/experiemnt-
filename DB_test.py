from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Replace placeholders with your objects
csv_path = "/Volumes/<catalog>/<schema>/<volume>/path/to/your_file.csv"

df = (
    spark.read
         .option("header", True)        # set False if no header row
         .option("inferSchema", True)   # or provide schemas explicitly for reliability
         .option("multiLine", True)     # only if rows can span multiple lines
         .option("escape", '"')         # helpful when quotes appear in fields
         .csv(csv_path)
)

df.printSchema()
df.show(10, truncate=False)


display(dbutils.fs.ls("/Volumes/<catalog>/<schema>/<volume>/path/to/"))

import pyspark.pandas as ps

csv_path = "/Volumes/<catalog>/<schema>/<volume>/path/to/your_file.csv"
psdf = ps.read_csv(csv_path)     # reads from the same path
psdf.head()

pdf = df.toPandas()   # only for smaller datasets that fit driver memory


(df.write
   .mode("overwrite")
   .option("header", True)
   .csv("/Volumes/<catalog>/<schema>/<volume>/out/your_file_csv"))
