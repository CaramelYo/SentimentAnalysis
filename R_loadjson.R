library(jsonlite)
library(magrittr)
library(dplyr)
library(ndjson)

# system.time(bitly01 <- ndjson::stream_in("/Users/royzhuang/data/twitter_db/filterstream1.json")) #

system.time(filterstream1 <- jsonlite::stream_in(file("/Users/royzhuang/data/twitter_db/filterstream1.json"), 
                                           verbose=FALSE, pagesize=10000))

paste("/Users/royzhuang/data/twitter_db/filterstream",1,".json", sep="")

filterstream <- list()

system.time( for ( i in 1:18) {
  twitter <- paste("/Users/royzhuang/data/twitter_db/filterstream",i,".json", sep="")
  filterstream[[i]] <- jsonlite::stream_in(file(twitter),verbose=FALSE, pagesize=50000)
} )


dplyr::glimpse(filterstream[[16]])

filterstream[[4]]$text
