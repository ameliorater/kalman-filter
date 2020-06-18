library(dplyr)
library(ggplot2)

df <- read.csv("~/go/src/kalman-filter/result.csv", na.strings = c("", "NA"))
attach(df)

xVals = c(dataX, measurementX, predictionX)
yVals = c(dataY, measurementY, predictionY)
Labels = c(rep("Real location", length(dataX)), rep("Sensor measurement", length(measurementX)), rep("Filter prediction", length(predictionX)))

newDf = data.frame(xVals, yVals, Labels)

newDf %>%
  ggplot(aes(x = xVals, y = yVals)) + 
  geom_line(aes(color = Labels, linetype = Labels)) +
  geom_point(aes(color = Labels)) +
  theme_classic(base_size=22, base_family="mono") +
  labs(x = "X Position", y = "Y Position", title = "Kalman Filter Position Tracking Example")
