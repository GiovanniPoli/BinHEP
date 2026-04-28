gen_data_SCENARIO_I   = function(seed){
  
  set.seed(seed)
  
  # gen data 
  eta_g1 = qnorm( c( rep(1  /60,  365),  rep( 2/365,  365)) )
  eta_g2 = qnorm( c( rep(2  /60,  365),  rep( 4/365,  365)) )
  eta_g3 = qnorm( c( rep(5  /365, 365),  rep(.5/365,  365)) )
  eta_g4 = qnorm( c( rep(10 /365, 365),  rep( 1/365,  365)) )
  
  
  G1_1 = t(sapply(1:25, function(x) rbinom(730, size = 1, prob = pnorm(eta_g1))))
  G1_2 = t(sapply(1:25, function(x) rbinom(730, size = 1, prob = pnorm(eta_g2))))
  G2_1 = t(sapply(1:25, function(x) rbinom(730, size = 1, prob = pnorm(eta_g3))))
  G2_2 = t(sapply(1:25, function(x) rbinom(730, size = 1, prob = pnorm(eta_g4))))
  
  G1 = rbind(G1_1,G1_2)
  G2 = rbind(G2_1,G2_2)
  
  for(i in 1:50){
    start_c = sample.int(723,1)
    end_c   = min(730, start_c + sample.int(23,1) + 7)
    
    G2[i, start_c:end_c] = rbinom(n = end_c- start_c+1 , size = 1, prob = 99/100)
  }
  
  raw_data = rbind(G1,G2)
  
  raw_data[rbinom(n = 730*100, size = 1, prob = .1)==1] = NA
  raw_list = lapply(1:100, function(x){
    s_and_end = where_cut_ts_ext_NA_sim(raw_data[x,]) 
    
    ts  = raw_data[x,s_and_end[1]:s_and_end[2]]
    Trt = c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z   = matrix(1, nrow = length(ts), ncol = 12)
    Z[,2] = (c(1:365,1:365)/365) [s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1",2] = Z[Trt == "Trt.1",2]+1/365 - (1:365/365)[s_and_end[1]]
    
    list("ts"    = ts, 
         "Trt"   = Trt,
         "Z"     = Z,
         "Tstar" = c(365+1-s_and_end[1], 3650000))
  })

  df_time_constant = data.frame( f1 = factor(c(rep(c("Lower risk", "Higher risk",
                                                     "Lower risk", "Higher risk"), each = 25))),
                                 f2 = factor(c(rep(c("Level 1","Level 2"), 50))))
  
  
  
  X = data.matrix(df_time_constant) - 1.5
  X = cbind(1,X)
  
  names(raw_list) =  rownames(X) = c(paste0(rep("G",100), rep(c("1","2"), each = 50),"_",1:100))
  
  sim_data =list( "df_time_constant" = df_time_constant,
                  "data_list"        = raw_list,
                  "X"                = X,
                  "seed"             = seed)
  sim_data
}
gen_data_SCENARIO_II  = function(seed){
  
  set.seed(seed)
  
  # gen data 
  eta_g1 = qnorm( c( rep(1  /60,  365),  rep( 2/365,  365)) )
  eta_g2 = qnorm( c( rep(2  /60,  365),  rep( 4/365,  365)) )
  eta_g3 = qnorm( c( rep(5  /365, 365),  rep(.5/365,  365)) )
  eta_g4 = qnorm( c( rep(10 /365, 365),  rep( 1/365,  365)) )
  
  
  G1_1 = t(sapply(1:45, function(x) rbinom(730, size = 1, prob = pnorm(eta_g1))))
  G1_2 = t(sapply(1:45, function(x) rbinom(730, size = 1, prob = pnorm(eta_g2))))
  G2_1 = t(sapply(1:5,  function(x) rbinom(730, size = 1, prob = pnorm(eta_g3))))
  G2_2 = t(sapply(1:5,  function(x) rbinom(730, size = 1, prob = pnorm(eta_g4))))
  
  G1 = rbind(G1_1,G1_2)
  G2 = rbind(G2_1,G2_2)
  
  for(i in 1:10){
    start_c = sample.int(723,1)
    end_c   = min(730, start_c + sample.int(23,1) + 7)
    
    G2[i, start_c:end_c] = rbinom(n = end_c- start_c+1 , size = 1, prob = 99/100)
  }
  
  raw_data = rbind(G1,G2)
  
  raw_data[rbinom(n = 730*100, size = 1, prob = .1)==1] = NA
  
  raw_list = lapply(1:100, function(x){
    s_and_end = where_cut_ts_ext_NA_sim(raw_data[x,]) 
    
    ts = raw_data[x,s_and_end[1]:s_and_end[2]]
    Trt = c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z = matrix(1, nrow = length(ts), ncol = 12)
    Z[,2] = (c(1:365,1:365)/365) [s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1",2] = Z[Trt == "Trt.1",2]+1/365 - (1:365/365)[s_and_end[1]]
    
    list("ts"  = ts, 
         "Trt" = Trt,
         "Z"   = Z,
         "Tstar" = c(365+1-s_and_end[1], 3650000))
  })
  
  
  df_time_constant = data.frame( f1 = factor(c(rep(c("Lower risk",
                                                     "Higher risk",
                                                     "Lower risk",
                                                     "Higher risk"), each = 25))),
                                 f2 = factor(c(rep(c("Level 1","Level 2"), 50))))
  
  
  
  X = data.matrix(df_time_constant) - 1.5
  X = cbind(1,X)
  names(raw_list) =  rownames(X) = 
    c(paste0(rep("G",100), rep(c("1","2"), each = 50),"_",1:100))
  
  sim_data =list( "df_time_constant" = df_time_constant,
                  "data_list" = raw_list,
                  "X" = X,
                  "seed" = seed)
  sim_data
}
gen_data_SCENARIO_III = function(seed){
  
  set.seed(seed)
  
  # gen data 
  eta_g1 = qnorm( c( rep(1  /60,  365),  rep( 2/365,  365)) )
  eta_g2 = qnorm( c( rep(2  /60,  365),  rep( 4/365,  365)) )
  eta_g3 = qnorm( c( rep(5  /365, 365),  rep(.5/365,  365)) )
  eta_g4 = qnorm( c( rep(10 /365, 365),  rep( 1/365,  365)) )
  
  
  G1_1 = t(sapply(1:5,  function(x) rbinom(730, size = 1, prob = pnorm(eta_g1))))
  G1_2 = t(sapply(1:5,  function(x) rbinom(730, size = 1, prob = pnorm(eta_g2))))
  G2_1 = t(sapply(1:45, function(x) rbinom(730, size = 1, prob = pnorm(eta_g3))))
  G2_2 = t(sapply(1:45, function(x) rbinom(730, size = 1, prob = pnorm(eta_g4))))
  
  G1 = rbind(G1_1,G1_2)
  G2 = rbind(G2_1,G2_2)
  
  for(i in 1:90){
    start_c = sample.int(723,1)
    end_c   = min(730, start_c + sample.int(23,1) + 7)
    
    G2[i, start_c:end_c] = rbinom(n = end_c- start_c+1 , size = 1, prob = 99/100)
  }
  
  raw_data = rbind(G1,G2)
  
  raw_data[rbinom(n = 730*100, size = 1, prob = .1)==1] = NA
  
  raw_list = lapply(1:100, function(x){
    s_and_end = where_cut_ts_ext_NA_sim(raw_data[x,]) 
    
    ts = raw_data[x,s_and_end[1]:s_and_end[2]]
    Trt = c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z = matrix(1, nrow = length(ts), ncol = 12)
    Z[,2] = (c(1:365,1:365)/365) [s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1",2] = Z[Trt == "Trt.1",2]+1/365 - (1:365/365)[s_and_end[1]] # correction
    
    list("ts"  = ts, "Trt" = Trt, "Z"   = Z,  "Tstar" = c(365+1-s_and_end[1], 3650000))
  })
  
  
  df_time_constant = data.frame( f1 = factor(c(rep(c("Lower risk", "Higher risk",
                                                     "Lower risk", "Higher risk"), each = 25))),
                                 f2 = factor(c(rep(c("Level 1","Level 2"), 50))))
  
  
  
  X = data.matrix(df_time_constant) - 1.5
  X = cbind(1,X)
  names(raw_list) =  rownames(X) = 
    c(paste0(rep("G",100), rep(c("1","2"), each = 50),"_",1:100))
  
  sim_data =list( "df_time_constant" = df_time_constant,
                  "data_list" = raw_list,
                  "X" = X,
                  "seed" = seed)
  sim_data
}
gen_data_SCENARIO_IV  = function(seed, G_par){
  
  set.seed(seed)
  
  # gen data 
  mu_g1 = -3
  mu_g2 = -2.5
  
  raw_data = matrix(NA, ncol = 730, nrow = 100)
  Z_all     = matrix(1, nrow = 730, ncol = 12)
  Z_all[,2] = (c(1:365,1:365)/365)
  
  for(i in 1:50){
    raw_data[i, ] = sample_biniary_ts_wSplit_from_G( mu_g1, TT     = 730, 
                                                     matFF  = Z_all,
                                                     mu0    = G_Par$m0, 
                                                     Sigma0 = G_Par$S0,
                                                     G      = G_Par$G1,
                                                     G_star = G_Par$G2,
                                                     SigmaEps      = G_Par$Ve, 
                                                     SigmaEps_star = G_Par$Ves,
                                                     Tstar         = c(366, 360000) )
  }
  for(i in 51:100){
    raw_data[i, ] = sample_biniary_ts_wSplit_from_G( mu_g2, TT     = 730, 
                                                     matFF  = Z_all,
                                                     mu0    = G_Par$m0, 
                                                     Sigma0 = G_Par$S0,
                                                     G      = G_Par$G1,
                                                     G_star = G_Par$G2,
                                                     SigmaEps      = G_Par$Ve, 
                                                     SigmaEps_star = G_Par$Ves,
                                                     Tstar         = c(366, 360000) )
  }
  raw_data[rbinom(n = 730*100, size = 1, prob = .1)==1] = NA
  
  raw_list = lapply(1:100, function(x){
    s_and_end = where_cut_ts_ext_NA_sim(raw_data[x,]) 
    
    ts    = raw_data[x,s_and_end[1]:s_and_end[2]]
    Trt   = c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z     = matrix(1, nrow = length(ts), ncol = 12)
    Z[,2] = (c(1:365,1:365)/365) [s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1",2] = Z[Trt == "Trt.1",2]+1/365 - (1:365/365)[s_and_end[1]]
    
    list("ts"  = ts,  "Trt" = Trt,
         "Z"   = Z,   "Tstar" = c(365+1-s_and_end[1], 3650000))
  })
  
  df_time_constant = data.frame( f1 = factor(c(rep(c("Lower risk", "Higher risk",
                                                     "Lower risk", "Higher risk"), each = 25))),
                                 f2 = factor(c(rep(c("Level 1","Level 2"), 50))))
  
  
  
  X = data.matrix(df_time_constant) - 1.5
  X = cbind(1,X)
  names(raw_list) =  rownames(X) = c(paste0(rep("G",100), rep(c("1","2"), each = 50),"_",1:100))
  
  sim_data =list( "df_time_constant" = df_time_constant,
                  "data_list" = raw_list, "X" = X, "seed" = seed)
  sim_data
}
