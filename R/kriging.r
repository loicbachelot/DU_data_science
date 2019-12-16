#rm(list = ls())
#gc()

#load the libraries
library(gstat)
library(sp)
library(spacetime)
library(raster)
library(rgdal)
library(rgeos) 

#Load the data
data <- read.table("data_ozon_tram1_14102011_14012012.csv", sep=",", header=T)

#changing the time format
data$TIME <- as.POSIXlt(as.numeric(substr(paste(data$generation_time), 1, 10)), origin="1970-01-01")

#changing the latitude and longitude format
data$LOG <- as.numeric(substr(paste(data$longitude),1,1))+(as.numeric(substr(paste(data$longitude),2,10))/60) 
data$LAT <- as.numeric(substr(paste(data$latitude),1,2))+(as.numeric(substr(paste(data$latitude),3,10))/60)
data <- na.omit(data)

#using a subset
sub <- data[data$TIME>=as.POSIXct('2011-12-12 00:00 CET')&data$TIME<=as.POSIXct('2011-12-14 23:00 CET'),]

#creat the spacial points objects
coordinates(sub) <- ~LOG+LAT
#get the origine
projection(sub)=CRS("+init=epsg:4326")

#Transform into meters and not degrees
ozone.UTM <- spTransform(sub,CRS("+init=epsg:3395")) 
ozoneSP <- SpatialPoints(ozone.UTM@coords,CRS("+init=epsg:3395"))

#removing duplicates
dupl <- zerodist(ozoneSP)
ozoneDF <- data.frame(PPB=ozone.UTM$ozone_ppb[-dupl[,2]]) 
ozoneTM <- as.POSIXct(ozone.UTM$TIME[-dupl[,2]],tz="CET")

#reassemble everything
timeDF <- STIDF(ozoneSP,ozoneTM,data=ozoneDF)
stplot(timeDF)


#variogram
#var <- variogramST(PPB~1,data=timeDF,tunit="hours",assumeRegular=F,na.omit=T)
plot(var,map=F)
plot(var, map=T)

#Different kriging

#initialisation 
# lower and upper bounds
pars.l <- c(sill.s = 0, range.s = 10, nugget.s = 0,sill.t = 0, range.t = 1, nugget.t = 0,sill.st = 0, range.st = 10, nugget.st = 0, anis = 0)
pars.u <- c(sill.s = 200, range.s = 1000, nugget.s = 100,sill.t = 200, range.t = 60, nugget.t = 100,sill.st = 200, range.st = 1000, nugget.st = 100,anis = 700)

#separable (not making much sense since we have a direct correlation between our space-time data)
separable <- vgmST("separable", space = vgm(-60,"Sph", 500, 1),time = vgm(35,"Sph", 500, 1), sill=0.56)
plot(var,separable,map=F) 

#getting the error
separable_Vgm <- fit.StVariogram(var, separable, fit.method=0)
attr(separable_Vgm,"MSE")


#Product sum
prodSumModel <- vgmST("productSum",space = vgm(1, "Exp", 150, 0.5),time = vgm(1, "Exp", 5, 0.5),k = 50)
prodSumModel_Vgm <- fit.StVariogram(var, prodSumModel,method = "L-BFGS-B",lower=pars.l)
attr(prodSumModel_Vgm, "MSE")
plot(var, prodSumModel_Vgm, map=F)


#Metric
metric <- vgmST("metric", joint = vgm(50,"Mat", 500, 0), stAni=2000) #weird error with a stAni at 200 I have a "stAni" must be positive."
metric_Vgm <- fit.StVariogram(var, metric, method="L-BFGS-B",lower=pars.l)
attr(metric_Vgm, "MSE")
plot(plot(var, metric_Vgm, map=F))


#Sum metric
sumMetric <- vgmST("sumMetric", space = vgm(psill=5,"Sph", range=500, nugget=0),time = vgm(psill=500,"Sph", range=500, nugget=0), joint = vgm(psill=1,"Sph", range=500, nugget=10), stAni=500) 
sumMetric_Vgm <- fit.StVariogram(var, sumMetric, method="L-BFGS-B",lower=pars.l,upper=pars.u,tunit="hours")
attr(sumMetric_Vgm, "MSE")
plot(var, sumMetric_Vgm, map=F)


#simple sum metric
SimplesumMetric <- vgmST("simpleSumMetric",space = vgm(5,"Sph", 500, 0),time = vgm(500,"Sph", 500, 0), joint = vgm(1,"Sph", 500, 0), nugget=1, stAni=500) 
SimplesumMetric_Vgm <- fit.StVariogram(var, SimplesumMetric,method = "L-BFGS-B",lower=pars.l)
attr(SimplesumMetric_Vgm, "MSE")


#comparing the models
plot(var,list(separable_Vgm, prodSumModel_Vgm, metric_Vgm, sumMetric_Vgm, SimplesumMetric_Vgm),all=T,wireframe=T) 



#Prediction

#creating the spatio-temporal prediction grid
plot(sub)
sp.grid.UTM <- ozoneSP #using the same roads
plot(sp.grid.UTM)
tm.grid <- seq(as.POSIXct('2011-12-12 06:00 CET'),as.POSIXct('2011-12-14 09:00 CET'),length.out=5)
grid.ST <- STF(sp.grid.UTM,tm.grid) 

#doing the prediction
pred <- krigeST(PPB~1, data=timeDF, modelList=sumMetric_Vgm, newdata=grid.ST)
stplot(pred)
