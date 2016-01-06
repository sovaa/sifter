from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

#sc = SparkContext(appName="PythonCollaborativeFilteringExample")
data = sc.textFile("/home/oscar/repo/git/sifter/mf/data-big.txt")
ratings = data.map(lambda l: l.split('\t')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

model.save(sc, "/home/oscar/repo/git/sifter/mf/data-big.sprk")
