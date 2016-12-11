library(dplyr)
library(lubridate)
library(BH)
taxiData <- read.csv('tripData.csv', stringsAsFactors = FALSE)

taxiData$tpep_pickup_datetime <- ymd_hms(taxiData$tpep_pickup_datetime)

taxiData$Day <- wday(taxiData$tpep_pickup_datetime)

tripsPlot <- function(day, startHour, endHour) {
  tripsData <- subset(taxiData, Day == day)
  tripsTable <- table(hour(tripsData$tpep_pickup_datetime))
  tripsTable <- as.table(tripsTable[(startHour + 1):(endHour + 1)])
  taxiPlot <- plot(tripsTable, type = 'o', xlab = 'Hour', xlim = c(startHour, endHour))
  return(taxiPlot)
}

shinyServer(
  function(input, output) {
    output$day <- renderPrint({input$day})
    output$startHour <- renderPrint({input$startHour})
    output$endHour <- renderPrint({input$endHour})
    output$taxiPlot <- renderPlot({tripsPlot(input$day, input$startHour, input$endHour)})
  }
)
