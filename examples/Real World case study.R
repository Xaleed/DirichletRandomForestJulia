

#
# Real-World case study for Dirichlet-random forest for compositional data
#
# Data from the SAPD - only topsoil soil texture
#
# Environmental covariates from SoilGrids2.0
#
# Authors of this script: S vd Westhuizen & Khaled Masoumifard 
# Stellenbosch University, 2025
#

# Load Packages

library(sf)
library(terra)
library(tmap)
library(maptiles)
library(geodata)
#library(OpenStreetMap)
library(RColorBrewer)
library(ranger)
library(DirichletReg)
library(ggplot2)
library(ggpattern)
library(compositions)
library(beepr)
library(parallel)
library(JuliaCall)
library(xtable)
library(Ternary)
library(reshape2)

#set wd
wd = getwd()

#Import data
dat = read.csv("d.csv", sep="")
dat = dat[,-ncol(dat)] #drop external CV index
dat = dat[,-81]  #drop cov name which is a duplicate


#set the projection for coordinates

Homolosine_crs = 'PROJCS["Homolosine", 
                     GEOGCS["WGS 84", 
                            DATUM["WGS_1984", 
                                  SPHEROID["WGS 84",6378137,298.257223563, 
                                           AUTHORITY["EPSG","7030"]], 
                                  AUTHORITY["EPSG","6326"]], 
                            PRIMEM["Greenwich",0, 
                                   AUTHORITY["EPSG","8901"]], 
                            UNIT["degree",0.0174532925199433, 
                                 AUTHORITY["EPSG","9122"]], 
                            AUTHORITY["EPSG","4326"]], 
                     PROJECTION["Interrupted_Goode_Homolosine"], 
                     UNIT["Meter",1]]'


#save composition names and covariate names
Ynames = colnames(dat)[5:7]
Xnames = colnames(dat)[8:147]

#Check composities
summary(apply(dat[,Ynames], 1, sum)) # majority of data dont sum to 100

#do a quick fix to run models
for(i in 1:nrow(dat)) {
  if(i==1) {d = dat[,c("x", "y", Ynames, Xnames)]}
  add = (100-sum(dat[i,Ynames]))/3
  d[i,Ynames] = dat[i,Ynames] + add
}

summary(d[,Ynames])
d$silt = ifelse(d$silt<=0, .01, d$silt)

#sum(apply(d[,Ynames], 1, sum) == 100)
#d[which(apply(d[,Ynames], 1, sum) != 100),][,Ynames]


#Create spatial object for mapping
sd = st_as_sf(d, coords=c("x", "y"), crs=Homolosine_crs)


#Download South Africa polygons
#sa_shp = rnaturalearth::ne_countries(country="South Africa")
sa_geodat = geodata::gadm("South Africa", path=paste0(getwd(),"/SAData"))
sa_shp = st_transform(st_as_sf(sa_geodat), crs(sd)) 
kzn_shp = sa_shp[sa_shp$NAME_1 == "KwaZulu-Natal",]


#Subset data only in KZN
pts_in_KZN = st_intersects(kzn_shp,sd)[[1]]
sd_KZN = sd[pts_in_KZN,]

#overwrite d
d = cbind(st_coordinates(sd_KZN),st_drop_geometry(sd_KZN[,c(Ynames,Xnames)]))


#point maps for paper


#tmap_mode("plot")
map_locs =  tm_basemap("CartoDB.Positron") +   # "Esri.WorldTopoMap"or "CartoDB.Positron" +
  tm_shape(sd_KZN) + tm_dots(fill="blue", fill_alpha = .5, size=.5) +
  tm_scalebar() +
  tm_compass() 


res=300
png(paste0(getwd(),"/Figures/Map_locations.png"), width=750/96*res, height=600/96*res, res=res)
map_locs
dev.off()


#Investigate compositional data


png(paste0(getwd(),"/Figures/Tern_plot.png"))
#tern_dens = TernaryDensity(d[,Ynames], resolution = 10L)
TernaryPlot(alab = "clay",blab = "silt",clab = "sand")
#ColourTernary(tern_dens, legend = TRUE, bty = "n", title = "Density")
TernaryPoints(d[,Ynames], pch=".", col="blue", cex=4)
dev.off()




############################################################################
#Import Environmental covariates from SG2
#directory for covs
setwd("F:\\Stephan\\Research\\BiplotXML\\Covariates\\South-Africa_250m\\")
stack = list.files(pattern="tif$", full.names=FALSE)
covs = terra::rast(stack)
rm(stack)
crs(covs) = crs(Homolosine_crs)
setwd(wd)

#Check categorical covariates and save to Xnames array (list is known from previous project)
Xnames_cat = names(covs)[c(54:83,85:98,99:116,118:126,168:171,173)]

#which cat names in d?
Xnames_cat = Xnames_cat[Xnames_cat %in% Xnames]

#Create numeric Xnames
Xnames_num = Xnames[!(Xnames %in% Xnames_cat)]

#Mask covs to shp
covs_masked = mask(covs,kzn_shp)
plot(covs_masked[[1]])    #plot one to check 

#analyse covs 
rm_Xnamescat_zero = names(which(apply(d[,Xnames_cat],2,function(x){length(unique(x))==1})))
rm_Xnamesnum_zero = names(which(apply(d[,Xnames_num],2,function(x){length(unique(x))==1}))) 

Xnames_cat = Xnames_cat[!(Xnames_cat %in% rm_Xnamescat_zero)]
Xnames_num = Xnames_num[!(Xnames_num %in% rm_Xnamesnum_zero)]
Xnamesf = c(Xnames_num,Xnames_cat) #final Xnames

#convert categorical variables to factors
#datx_fac = d[,Xnames_cat]
#datx_fac[,Xnames_cat] = lapply(d[,Xnames_cat],as.factor)

#create modelling data set, d
#d_dir = d
#d = cbind(d[,c("X","Y",Ynames,Xnames_num)],datx_fac)
#d = d[complete.cases(d),] #remove rows where covs are missing    #none removed
#colnames(d)[1:2] = c("x", "y")


#######################################################################
#Now that Xnames have been finalised, we can create the covs data frame

#create covs data set - will be used for prediction to create maps
covs_df = terra::as.data.frame(covs_masked, xy=T)
#Convert categorical variables to factors
#covs_df_fac = covs_df[,Xnames_cat]
#covs_df_fac[,Xnames_cat] = lapply(covs_df[,Xnames_cat],as.factor)
#covs_df_final = cbind(covs_df[,c("x","y",Xnames_num)],covs_df_fac)

covs_df_final = cbind(covs_df[,c("x","y",Xnamesf)])

#some covariates may have many NA values and need to be removed
#check_covs_cat = apply(covs_df_final[,Xnames_cat],2,function(x) {sum(is.na(x))}) #not a problem
#check_covs_num = apply(covs_df_final[,Xnames_num],2,function(x) {sum(is.na(x))}) # not a problem
#Xnames_cat_rm = names(which(check_covs_cat>1e06))
#Xnames_num_rm = names(which(check_covs_num>1e06))

#Use Xnames to map
covs_df_final2 = covs_df_final[complete.cases(covs_df_final),]

#redefine factors in d and in covs_df_final2
#for(i in 1:length(Xnames_cat)){
#  if(length(levels(d[,Xnames_cat[i]]))!=length(levels(covs_df_final2[,Xnames_cat[i]]))){
#    if(length(levels(d[,Xnames_cat[i]]))>length(levels(covs_df_final2[,Xnames_cat[i]]))){
#      levels(covs_df_final2[,Xnames_cat[i]]) = levels(d[,Xnames_cat[i]])
#    } else {levels(d[,Xnames_cat[i]]) = levels(covs_df_final2[,Xnames_cat[i]])}
#  } 
#}


#remove unnecessary objects to save up space
rm(covs,covs_df_fac,covs_df)
gc()



#################################################
##  Modelling
#################################################

#steps for Dirichlet RF

#Download Julia, and install Julia on your PC

#install these packages in Julia
#} add Random, Distributions, CSV, DataFrames, StatsBase, Statistics, SpecialFunctions, Optim, NLopt, ForwardDiff, BenchmarkTools

#install JuliaCall in R

# Initialize Julia (if not working locate the binary folder)
julia_setup("C:\\Users\\stephanvdw\\AppData\\Local\\Programs\\Julia-1.11.3\\bin")

# Load required Julia packages
julia_command("using Random, Distributions, CSV, DataFrames, StatsBase, Statistics, 
              SpecialFunctions, Optim, NLopt, ForwardDiff, BenchmarkTools")

# Source Julia code
julia_source("F:\\Stephan\\Research\\DirichletRF\\DirichletRandomForestJulia-master\\DirichletRandomForestJulia-master\\src\\dirichlet_forest_ml.jl")
julia_source("F:\\Stephan\\Research\\DirichletRF\\DirichletRandomForestJulia-master\\DirichletRandomForestJulia-master\\src\\rf_for_call_to_R.jl")
julia_source("F:\\Stephan\\Research\\DirichletRF\\DirichletRandomForestJulia-master\\DirichletRandomForestJulia-master\\src\\dirichlet_forest_ml_distributed.jl")
julia_source("F:\\Stephan\\Research\\DirichletRF\\DirichletRandomForestJulia-master\\DirichletRandomForestJulia-master\\src\\distribute_forest.jl")




#Load functions used in the modelling

evaluate_performance = function(Y_true, Y_pred) {
  n_samples = nrow(Y_true)
  
  aitchison_dist <- mean(sapply(1:n_samples, function(i) {
    y_true = as.numeric(Y_true[i,])
    y_pred = as.numeric(Y_pred[i,])
    
    y_true = pmax(y_true, .Machine$double.eps)
    y_pred = pmax(y_pred, .Machine$double.eps)
    
    gm_true = geometric_mean(y_true)
    gm_pred <- geometric_mean(y_pred)
    
    sqrt(sum((log(y_true/gm_true) - log(y_pred/gm_pred))^2))
  }))
  
  total_var = sum(sapply(1:n_samples, function(i) {
    y_true = as.numeric(Y_true[i,])
    y_true = pmax(y_true, .Machine$double.eps)
    gm_true = geometric_mean(y_true)
    sum(((y_true))^2)
  }))
  
  residual_var <- sum(sapply(1:n_samples, function(i) {
    y_true = as.numeric(Y_true[i,])
    y_pred = as.numeric(Y_pred[i,])
    
    y_true = pmax(y_true, .Machine$double.eps)
    y_pred = pmax(y_pred, .Machine$double.eps)
    
    gm_true = geometric_mean(y_true)
    gm_pred = geometric_mean(y_pred)
    
    sum(((y_true) - (y_pred))^2)
  }))
  
  comp_r2 = 1 - residual_var/total_var
  mse = colMeans((Y_true - Y_pred)^2)
  rmse = sqrt(mse)
  
  return(list(
    aitchison_distance = aitchison_dist,
    MEC = comp_r2,
    RMSE = rmse,
    mean_rmse = sqrt(mean(mse))
  ))
}


#   Prepare for modelling
###########################################
#set modelling parameters

#k-fold cross-validation (only testing - not going to calibrate - will use default parameter values)
k = 10

#modelling data set
dm = d[,c(Ynames, Xnamesf)]


#create index for folds in the cross-validation
set.seed(25468)
dm = dm[sample(1:nrow(dm)),]
dm$cv_inds = cut(seq(1,nrow(dm)),breaks=k,labels=FALSE)
#re-order the data set 
dm = dm[order(as.numeric(row.names(dm))),]
cv_nr = ncol(dm)

#functions for soil texture for models
#fn_silt = as.formula(paste0("silt~",paste(Xnames,collapse="+")))
#fn_sand = as.formula(paste0("sand~",paste(Xnames,collapse="+")))
#fn_clay = as.formula(paste0("clay~",paste(Xnames,collapse="+")))
#



####################################

loop_time = Sys.time()

for(l in 1:10){     #START CV LOOP
  
  if(l==1) {preds_out = vector(mode="list", length=k)}
  
  trn = dm[dm$cv_inds!=l,][,-cv_nr]
  tst = dm[dm$cv_inds==l,][,-cv_nr]
  
  x_trn = trn[,Xnamesf]
  y_trn = trn[,Ynames]
  x_tst = tst[,Xnamesf]
  y_tst = tst[,Ynames]
  
 
  #######################################
  # Dirichlet (Parametric) regression
  
  #data too big/complex/high-dimensional for this model
  
 # trn_dir = trn[,c(Xnamesf, Ynames)]
#  trn_dir$Y = DR_data(trn_dir[,Ynames])
#  fn_dir = as.formula(paste0("Y~",paste(Xnamesf,collapse="+")))
#  dir_reg_out = DirichReg(fn_dir, data=trn_dir, model="common")
  
  
  #######################################
  # CLR and ILR transformations with Random forests
  
  y_trn_clr = clr(as.matrix(y_trn))
  y_tst_clr = clr(as.matrix(y_tst))
  
  y_trn_ilr = ilr(as.matrix(y_trn))
  y_tst_ilr = ilr(as.matrix(y_tst))
  
  
  clr_models = lapply(1:ncol(y_trn_clr), function(i) {
    fn = as.formula(paste0(Ynames[i],"~",paste(Xnamesf,collapse="+")))
    clr_trn = cbind(y_trn_clr[,Ynames[i]], x_trn)
    colnames(clr_trn)[1] = Ynames[i]
    ranger(fn,clr_trn)
  })
  
  clr_tst_pred = do.call(cbind, lapply(clr_models, function(mod) predict(mod, as.data.frame(x_tst))$predictions))
  clr_tst_pred_inv = clrInv(clr_tst_pred)
  
  
  ilr_models = lapply(1:ncol(y_trn_ilr), function(i) {
    fn = as.formula(paste0("R",i,"~",paste(Xnamesf,collapse="+")))
    ilr_trn = cbind(y_trn_ilr[,i], x_trn)
    colnames(ilr_trn)[1] = paste0("R",i)
    ranger(fn, ilr_trn)
  })
  
  ilr_tst_pred = do.call(cbind, lapply(ilr_models, function(mod) predict(mod, as.data.frame(x_tst))$predictions))
  ilr_tst_pred_inv = ilrInv(ilr_tst_pred)
  
  rf_models = lapply(1:ncol(y_trn), function(i) {
    fn = as.formula(paste0("R",i,"~",paste(Xnamesf,collapse="+")))
    rf_trn = cbind(y_trn[,i], x_trn)
    colnames(rf_trn)[1] = paste0("R",i)
    ranger(fn, rf_trn)
  })
  rf_tst_pred = do.call(cbind, lapply(rf_models, function(mod) predict(mod, as.data.frame(x_tst))$predictions))
  
  
  
  
  
  #######################################
  # Dirichlet RF
  
  x_tst_m = as.matrix(x_tst)
  x_trn_m = as.matrix(x_trn)
  y_tst_m = as.matrix(y_tst)
  y_trn_m = as.matrix(y_trn)
  
  julia_assign("x_tst_m", x_tst_m)
  julia_assign("x_trn_m", x_trn_m)
  julia_assign("y_tst_m", y_tst_m)
  julia_assign("y_trn_m", y_trn_m)
  
  
  
  
  
  # Run parallel training
  time_taken = system.time({
    julia_pred = julia_eval('begin 
    
    # Process input data
    x_trn_m, y_trn_m, x_tst_m = process_matrix_data(x_trn_m, y_trn_m, x_tst_m)
    
    # Store processed data for later use
    global x_train_processed = x_trn_m
    global x_test_processed = x_tst_m
    
    # Initialize forest
    forest = DirichletForest(500)
    
    # Train forest using MLE-Newton method with parallel implementation
    println("Training forest in parallel...")
    fit_dirichlet_forest_parallel!(
      forest, 
      x_trn_m, 
      y_trn_m, 
      500000000,  
      200000,  
      5,   # min_node_size
      nothing,   # mtry -> nothing = default
      estimate_parameters_mle_newton
    )
    
    "Forest training completed"
  end')
  })
  
  
  
  
  #do not use below (this runs the code sequentially) - takes too long
  
 # rf_dir_time = Sys.time()
#  julia_pred = julia_eval('begin 
#               x_trn_m, y_trn_m, x_tst_m = process_matrix_data(x_trn_m, y_trn_m, x_tst_m)
#
#               # Initialize forest
#                forest = DirichletForest(500)
#
#                # Train forest using MLE-Newton method
#                println("Training forest...")
#                fit_dirichlet_forest!(
#                  forest, 
#                  x_trn_m, 
#                  y_trn_m, 
#                  500000000, 
#                  200000,   #max.depth
#                  5,        #min.node.size
#                  nothing,  #nothing = p/3
#                  estimate_parameters_mle_newton
#                )
#                end')
#  
#  rf_dir_time = Sys.time() - rf_dir_time



# Predict on tst data
pred_test = julia_eval('
    predict_dirichlet_forest(forest, x_tst_m)
')

# Predict on trn data (if needed)
#pred_train = julia_eval('
#    predict_dirichlet_forest(forest, x_trn_m)
#')

##################################################
#add all predictions to the output list

out = as.data.frame(cbind(clr_tst_pred_inv, ilr_tst_pred_inv, rf_tst_pred, pred_test))
colnames(out) = c(paste0("CLR_RF", Ynames), paste0("ILR_RF", Ynames), paste0("RF", Ynames), paste0("Dir_RF", Ynames))
preds_out[[l]] = out

  
}
loop_time = Sys.time() - loop_time


out_df = do.call(rbind, preds_out)



CLR_perf = evaluate_performance(d[,Ynames], out_df[,1:3]*100)
ILR_perf = evaluate_performance(d[,Ynames], out_df[,4:6]*100)
RF_perf = evaluate_performance(d[,Ynames], out_df[,7:9])
DIR_RF_perf = evaluate_performance(d[,Ynames], out_df[,10:12]*100)


perf_df = data.frame(RF_CLR=unlist(CLR_perf),
                     RF_ILR=unlist(ILR_perf),
                     RF=unlist(RF_perf),
                     DIR_RF=unlist(DIR_RF_perf),
                     row.names=c("aitchison_distance", "MECcomp", paste0(Ynames, "RMSE"), "meanRMSE"))


xtable(perf_df)






##################################################
# Produce maps with Dir-RF


x_trn = d[,Xnamesf]
y_trn = d[,Ynames]
x_tst = covs_df_final2[,Xnamesf]



x_trn_m = as.matrix(x_trn)
y_trn_m = as.matrix(y_trn)
x_tst_m = as.matrix(x_tst)

julia_assign("x_tst_m", x_tst_m)
julia_assign("x_trn_m", x_trn_m)
julia_assign("y_trn_m", y_trn_m)



# Run parallel training
time_taken = system.time({
  julia_pred = julia_eval('begin 
    
    # Process input data
    x_trn_m, y_trn_m, x_tst_m = process_matrix_data(x_trn_m, y_trn_m, x_tst_m)
    
    # Store processed data for later use
    global x_train_processed = x_trn_m
    global x_test_processed = x_tst_m
    
    # Initialize forest
    forest = DirichletForest(500)
    
    # Train forest using MLE-Newton method with parallel implementation
    println("Training forest in parallel...")
    fit_dirichlet_forest_parallel!(
      forest, 
      x_trn_m, 
      y_trn_m, 
      500000000,  
      200000,  
      5,   # min_node_size
      nothing,   # mtry
      estimate_parameters_mle_newton
    )
    
    "Forest training completed"
  end')
})



# Predict on tst data
pred_test = julia_eval('
    predict_dirichlet_forest(forest, x_tst_m)
')

map_preds = covs_df_final2[,c("x", "y")]
map_preds = cbind(map_preds, pred_test*100)
colnames(map_preds)[3:5] = Ynames

map_rstrs = terra::rast(map_preds,type="xyz",crs=terra::crs(Homolosine_crs))

brks = seq(0,100,by=10) #not really necessary with continuous scales


summary(map_preds[,Ynames]) 



map_sand = tm_shape(kzn_shp) + tm_borders() + tm_layout(frame=F, bg.color="white") +
  tm_shape(map_rstrs[["sand"]]) + tm_raster(col="sand", col.scale=
                       #     tm_scale_intervals(values = brewer.pal(length(brks),name="PiYG"),
                            tm_scale_continuous(values = brewer.pal(length(brks),name="PiYG")
                                                #,
                                                #midpoint = 45
                                      #  style="fixed",
                                      #  breaks=brks,
                                      #  labels=as.character(brks)[-1]
                                      ),
                         col.legend = tm_legend(title="Sand (%)", reverse=T))


map_silt = tm_shape(kzn_shp) + tm_borders() + tm_layout(frame=F, bg.color="white") +
  tm_shape(map_rstrs[["silt"]]) + tm_raster(col="silt", col.scale=
                                             # tm_scale_intervals(values = brewer.pal(length(brks),name="BrBG"),
                                             tm_scale_continuous(values = brewer.pal(length(brks),name="BrBG")
                                                                 #,
                                                                 #midpoint = 25
                                                            #     style="fixed",
                                                            #     breaks=brks,
                                                            #     labels=as.character(brks)[-1]
                                                            ),
                                            col.legend = tm_legend(title="Silt (%)", reverse=T))


map_clay = tm_shape(kzn_shp) + tm_borders() + tm_layout(frame=F, bg.color="white") +
  tm_shape(map_rstrs[["clay"]]) + tm_raster(col="clay", col.scale=
                                             # tm_scale_intervals(values = brewer.pal(length(brks),name="PuOr"),
                                              tm_scale_continuous(values = brewer.pal(length(brks),name="PuOr")
                                                                  #,
                                                                  # style="fixed",
                                                         #        midpoint = 35
                                                                 #,
                                                              #   breaks=brks,
                                                              #   labels=as.character(brks)[-1]
                                                               ),
                                            col.legend = tm_legend(title="Clay (%)", reverse=T))

###########################
#save maps
res = 300 #image res

png(paste0(wd,"/Figures/Map_sand.png"), width=750/96*res, height=600/96*res, res=res)
map_sand
dev.off()

png(paste0(wd,"/Figures/Map_silt.png"), width=750/96*res, height=600/96*res, res=res)
map_silt
dev.off()

png(paste0(wd,"/Figures/Map_clay.png"), width=750/96*res, height=600/96*res, res=res)
map_clay
dev.off()

#############################################
#Ternary plot for predictions

#png(paste0(getwd(),"/Figures/Tern_preds_plot.png"))
#TernaryPlot(alab = "clay",blab = "silt",clab = "sand")
#TernaryPoints(map_preds[,Ynames], pch=".", col="blue", cex=2)
#dev.off()


#png(paste0(getwd(),"/Figures/Tern_preds_plot_dens.png"))
#tern_dens = TernaryDensity(map_preds[,Ynames], resolution = 10L)
#TernaryPlot(alab = "clay",blab = "silt",clab = "sand")
#ColourTernary(tern_dens, legend = TRUE, bty = "n", title = "Density")
#dev.off()


#################################################
#Density plots for distributions for predictions


dens_dat = melt(map_preds[,Ynames])
colnames(dens_dat) = c("Composite", "Value")



ggplot(dens_dat, aes(x=Value, fill=Composite)) +
  scale_fill_manual(values=c("darkred", "darkblue", "darkgreen")) +
  geom_density_pattern(pattern_color = "white",
                       pattern_fill = "black",
                       alpha = 0.25,
                       bw=5,
                       pattern_alpha = 0.6,
                       aes(pattern = Composite)) + theme_bw() + 
  scale_pattern_manual(values=c(silt="none",clay="circle",sand="stripe")) +
  theme(
    axis.title.x = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank())





############################
#
#    END of SCRIPT
#
############################