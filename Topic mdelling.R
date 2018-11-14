library("tm")
library( doParallel )
library("wordcloud")
library("slam")
library("topicmodels")
library(tidyverse) # general utility & workflow functions
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(tm) # general text mining functions, making document term matrixes
library(SnowballC) # for stemming


Memo=read.csv('path/to/memos.csv')




#-----------------first part data processing concern only my own dataset ! 
Memo$DATEMEMOs=as.Date(as.POSIXct(Memo$DATEMEMO, origin = "1582-10-14"))
# View(head(Memo,100))
MemoAll= Memo 
Memo=subset(Memo,
            DATEMEMOs>as.Date("2016-31-12", format="%Y-%d-%m") 
            & 
              DATEMEMOs < as.Date("2017-31-12", format="%Y-%d-%m"))
Memo=Df
memos=Memo$TEXTE
##----------------------

n=round(length(memos)*0.2)
memos=memos[1:n]

# Clean Memos

memos = gsub("(RT|via)((?:\\b\\W*@\\w+)+)","",memos)
memos = gsub("http[^[:blank:]]+", "", memos)
memos = gsub("@\\w+", "", memos)
memos = gsub("[ \t]{2,}", "", memos)
memos = gsub("^\\s+|\\s+$", "", memos)
memos <- gsub('\\d+', '', memos)
memos = gsub("[[:punct:]]", " ", memos)
corpus = Corpus(VectorSource(memos))
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,stripWhitespace)
corpus = tm_map(corpus,tolower)
corpus = tm_map(corpus,removeWords,stopwords("french"))
tdm = DocumentTermMatrix(corpus) # Creating a Term document Matrix


# create tf-idf matrix
term_tfidf <- tapply(tdm$v/row_sums(tdm)[tdm$i], tdm$j, mean) * log2(nDocs(tdm)/col_sums(tdm > 0))
summary(term_tfidf)
tdm <- tdm[,term_tfidf >= 0.1]
tdm <- tdm[row_sums(tdm) > 0,]
summary(col_sums(tdm))

unique_indexes <- unique(tdm$i) # get the index of each unique value
tdm <- tdm[unique_indexes,] # get a subset of only those indexes



library(memisc)

split = detectCores()
eachStart = 25

cl = makeCluster(split)

registerDoParallel( cl )


# now start the chains
nchains <- 16
my_k <- 6 ## or a vector with 16 elements

results_list <- foreach::foreach(i=1:16 , 
                        .packages = c('topicmodels') %dopar% {
                          
                          LDA(tdm,6 )
                          
                          }, .progress = "text")


cl = makeCluster(detectCores())
registerDoParallel(cl)
x = foreach(i=1:detectCores(), .combine="c", .packages="topicmodels") %dopar%  LDA(tdm,6 )
stopCluster(cl)
x

Df=Memo[1:round(nrow(Memo)*0.01),]

Df$DATEMEMOs=NULL

write.csv(Df,"Junk of Memo for Kernel.csv")


#Deciding best K value using Log-likelihood method
best.model <- lapply(50, function(d){LDA(tdm, d)})
best.model.logLik <- as.data.frame(as.matrix(lapply(x, logLik)))


#calculating LDA
k = 6;#number of topics
SEED = 786; # number of memos used
CSC_TM <-list(VEM = LDA(tdm, k = k, control = list(seed = SEED)),
              VEM_fixed = LDA(tdm, k = k,control = list(estimate.alpha = FALSE, seed = SEED)),
              Gibbs = LDA(tdm, k = k, method = "Gibbs",control = list(seed = SEED, burnin = 1000,thin = 100, iter = 1000)),
              CTM = CTM(tdm, k = k,control = list(seed = SEED,var = list(tol = 10^-4), em = list(tol = 10^-3))))



#To compare the fitted models we first investigate the values of the models fitted with VEM and estimated and with VEM and fixed
sapply(CSC_TM[1:2], slot, "alpha")
sapply(CSC_TM, function(x) mean(apply(posterior(x)$topics, 1, function(z) - sum(z * log(z)))))
Topic <- topics(CSC_TM[["VEM"]], 1)
Terms <- terms(CSC_TM[["VEM"]], 30)
Terms


