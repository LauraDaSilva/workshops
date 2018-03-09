
# Michael Hahsler 
# https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf

# Experimenting with recommendation systems
# https://www.r-bloggers.com/testing-recommender-systems-in-r/

library(recommenderlab)
library(ggplot2)
library(dplyr)
library(readr)

# Load the data
data()
data(MovieLense)

# Data sets in package 'recommenderlab':
#     
# Jester5k                                       Jester dataset (5k sample)
# JesterJokes (Jester5k)                         Jester dataset (5k sample)
# MSWeb                                          Anonymous web data from www.microsoft.com
# MovieLense                                     MovieLense Dataset (100k)
# MovieLenseMeta (MovieLense)                    MovieLense Dataset (100k)

MovieLense #943 x 1664 rating matrix of class 'realRatingMatrix' with 99392 ratings
?realRatingMatrix # A matrix containing ratings (typically 1-5 stars, etc.).
?MovieLense 
#The 100k MovieLense ratings data set collected through the MovieLens web site 
#from Sep 1997 and April 1998.It contains about 100,000 ratings (1-5) from 943 users on 1664 movies.
#It is an object of class "realRatingMatrix"

## If we wanted to convert this data into DF
movielenseTable <- as(MovieLense[1,],"list")
movieDf <- as.data.frame(movielenseTable[1])
movieDf$MovieName <- rownames(movieDf)
names(movieDf) <- c("rating", "MovieName")
row.names(movieDf)<-NULL

#### Exploration of the dataset ####

## visualize part of the matrix
recommenderlab::image(MovieLense[1:100,1:100])

## number of ratings per user
hist(recommenderlab::rowCounts(MovieLense))

## number of ratings per movie
hist(recommenderlab::colCounts(MovieLense))

## mean rating (averaged over users)
mean(recommenderlab::rowMeans(MovieLense))

## available movie meta information
head(MovieLenseMeta)

# Visualizing a sample of this
recommenderlab::image(sample(MovieLense, 500), main = "Raw ratings")

# Visualizing ratings
ggplot2::qplot(recommenderlab::getRatings(MovieLense), binwidth = 1, 
      main = "Histogram of ratings", xlab = "Rating")

summary(recommenderlab::getRatings(MovieLense)) # Skewed to the right
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.00    3.00    4.00    3.53    4.00    5.00


# How about after normalization?
ggplot2::qplot(recommenderlab::getRatings(normalize(MovieLense, method = "Z-score")),
      main = "Histogram of normalized ratings", xlab = "Rating") 

summary(recommenderlab::getRatings(normalize(MovieLense, method = "Z-score"))) # seems better
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -4.8520 -0.6466  0.1084  0.0000  0.7506  4.1280

# How many movies did people rate on average
ggplot2::qplot(recommenderlab::rowCounts(MovieLense), binwidth = 10, 
      main = "Movies Rated on average", 
      xlab = "# of users", 
      ylab = "# of movies rated")
# Seems people get tired of rating movies at a logarithmic pace. But most rate some.

# What is the mean rating of each movie
ggplot2::qplot(recommenderlab::colMeans(MovieLense), binwidth = .1, 
      main = "Mean rating of Movies", 
      xlab = "Rating", 
      ylab = "# of movies")

# The big spike on 1 suggests that this could also be intepreted as binary
# In other words, some people don't want to see certain movies at all.
# Same on 5 and on 3.
# We will give it the binary treatment later


#### Recommendations ###
recommenderlab::recommenderRegistry$get_entries(dataType = "realRatingMatrix")
# We have a few options:
# Alternating Least Squares (ALS_explicit): Explicit ratings based on latent factors (Collaborative Filtering)
# Alternating Least Squares (ALS_implicit): Implicit data based on latent factors (Collaborative Filtering)
# Item-Based Collaborative Filtering (IBCF)
# POPULAR: recommender based on item popularity
# RANDOM: Random recommendations
# RERECOMMEND: Re-recommends highly rated items
# SVD approximation with colum-mean imputation
# SVDF: SVD approximation with gradient descend
# UBCF: User-based collaborative filtering


# Let's check some algorithms against each other
scheme <- recommenderlab::evaluationScheme(MovieLense, method = "split", train = .9,
                           k = 1, given = 10, goodRating = 4)

scheme

algorithms <- list(
    "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
    "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
    "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
                                                   method="Cosine",
                                                   nn=50, minRating=3)),
    "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score"
    ))
    
)

# Run algorithms, predict next n movies
resultsTopList <- recommenderlab::evaluate(scheme, algorithms, type = "topNList", n=c(1, 3, 5, 10, 15, 20))
resultsTopList$`random items`

### TopList

# Draw ROC curve
recommenderlab::plot(resultsTopList, annotate = 1:4, legend="topleft")

# See precision / recall
# Precision = correctly recommended items/total recommended items
# Recall = correctly recommended items/total useful recommendations
recommenderlab::plot(resultsTopList, "prec/rec", annotate=3)

# Confusion matrices
recommenderlab::getConfusionMatrix(x=resultsTopList$`random items`)
recommenderlab::getConfusionMatrix(x=resultsTopList$`user-based CF`)
recommenderlab::getConfusionMatrix(x=resultsTopList$`item-based CF`)
recommenderlab::getConfusionMatrix(x=resultsTopList$`popular items`)
