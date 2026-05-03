#### Data cleaning function
library(stringr)
library(readr)
  
where_cut_ts_ext_NA_sim = function(ts){
    size  = length(ts)
    start = which.min(is.na(ts))
    end   = size+1-which.min(is.na(rev(ts)))
    return( c(start,end))
}
  
## TS
where_cut_ts_ext_NA = function(ts){
  size  = length(ts)
  start = which.min(is.na(ts))
  end   = size+1-which.min(is.na(rev(ts)))
  return(start:end)
}
## Trt.
change_last_values     = function(vec, value){
  size = length(vec)
  if(vec[size] != value) {
    return(vec)
  }
  if(sum(vec==value)==size){
    return(vec)
  }
  last_value = max(which(vec != value))
  vec[(last_value + 1):size] = vec[last_value]
  return(vec)
}
remove_first_if_unique = function(vec){
  c(ifelse(vec[1] != vec[2], FALSE, TRUE), rep(TRUE, length(vec)-1))
}
sub_seq_inside         = function(vec, len) {
  
  size = length(vec)
  length_of_seq = 1
  start_of_seq = 1
  
  for(i in 2:size){
    if(vec[i] == vec[i-1]){
      length_of_seq = length_of_seq + 1
    }else{
      if(length_of_seq < len){
        # Short inner seq are considered start of the new Trt.
        vec[start_of_seq:(i-1)]= vec[i]
      }
      start_of_seq  = i
      length_of_seq = 1
    }
  }
  
  return(vec)
}
## T*
set_star_times         = function(trt){
  size = length(trt)
  star_set  = c()
  for(i in 2:size){
    if(trt[i] != trt[i-1]) star_set = c(star_set,i)
  }
  return(star_set)
}

select_ts = function(data, meta_data, IDs, trt_index = seq(from = 8, to = 34, by = 2), verb = FALSE){
  
  trt_names = colnames(data)[trt_index]
  trt_names = stringr::str_remove(trt_names, "YN.")
  
  subject_to_check = NULL
  subject_exclued  = NULL
  
  data_list = list()
  
  for(subject in IDs){
    data_subject = data[data$subject_id == subject,]
    ts = data_subject$seiz_day_binary
    index = where_cut_ts_ext_NA(ts)
    ts = ts[index]
    
    set.seed(14)
    
    if(length(ts)> 30){
       data_subject = data[data$subject_id == subject,]
       ts = data_subject$seiz_day_binary

       full_ts_size = nrow(data_subject)
       
       ## Remove NA at start and end
       # c(NA,1,NA,0,NA) => c(1,NA,0)
       index = where_cut_ts_ext_NA(ts)
       ts = ts[index]
       
       # rowSums(trt_table)
       trt_table = data_subject[index, trt_index] == 1
       trt_1 = apply(trt_table,1, function(x) paste(trt_names[x], collapse = "_")) 
       
       ## Change Trt. Unkwon at the END 
       # c("A","Unkown","C","Unkown") => c("A","Unkown","C","C")
       trt_1["NA_NA_NA_NA_NA_NA_NA_NA_NA_NA_NA_NA_NA_NA" == trt_1] = "Unkown"
       trt_1 = change_last_values(trt_1,"Unkown")
       
       ## Remove 1 obs if unique at the start
       index = remove_first_if_unique(trt_1)
       ts = ts[index]
       trt_1 = trt_1[index]

       ## Impute short-inner seq as start of new TrT || Cut Off value
       # c("A",..,"A","B_A","B") =>  c("A",..,"A","B","B") 
       trt_1 = sub_seq_inside(trt_1,7)
       
       Tstar_i = c(set_star_times(trt_1), 365000)
       one_and_Tstar_i = c(1,Tstar_i)
       z_2i = vector(mode = "numeric", length(ts) )
       
       for(pos in 1:length(Tstar_i)){
         p0 = one_and_Tstar_i[pos]
         p1 = min( one_and_Tstar_i[pos+1],  length(ts)) 
         z_2i[ p0 : p1 ] = seq(from = 0,
                               to   = (p1-p0)/365,
                               length.out = p1-p0+1)
       }
       
     

       Z_i = matrix(1, ncol = 12, nrow = length(ts))
       Z_i[,2] = z_2i
       
       if(verb){
         cat(subject,":\nNumber of records:",length(ts[index]),"\n",
             "Length of the time-series:",length(ts[index]),"\n",
             "Number of Trt.:",length(unique(trt_1)),"Slices length:",table(trt_1),"\n",
             "Trt.Names:",unique(trt_1),"\n",
             "Check:",length(ts),length(trt_1), length(z_2i),"\n")
       }
     
       
       data_list[[subject]] = list( "ts"    = ts,
                                    "Trt"   = trt_1,
                                    "Z"     = Z_i,
                                    "Tstar" = Tstar_i
                                    )
    
    
       
    }else{
      subject_exclued = c(subject_exclued,subject)
    }
  
  
  }
  
  df_x = meta_data[ meta_data$subject_id %in% names(data_list),
                    c("sex",
                      "age",
                      "abnormal_mri",
                      "injury",
                      "family_history_of_seizures")]
  
  df_x = data.frame(df_x)
  df_x$sex = as.factor(df_x$sex)
  df_x$age = as.numeric(df_x$age)
  df_x$injury = as.factor(df_x$injury)
  df_x$abnormal_mri = as.factor(df_x$abnormal_mri) 
  df_x$family_history_of_seizures = as.factor(df_x$family_history_of_seizures)
  
  X = matrix(NA, nrow = nrow(df_x), ncol = ncol(df_x) + 1)
  colnames(X) = c("int.","sex","age","mri","injury" ,"history_seizures")
  rownames(X) = meta_data$subject_id[meta_data$subject_id %in% names(data_list)]
  X[,"int."] = 1 
  X[,"age"]  = df_x$age
  X[df_x$sex == "Female","sex"]  = -.5 
  X[df_x$sex == "Male",  "sex"]  =  .5 
  X[df_x$abnormal_mri == "No",      "mri"] = -.5
  X[df_x$abnormal_mri == "Unknown", "mri"] =   0
  X[df_x$abnormal_mri == "Yes",     "mri"] =  .5
  X[df_x$injury == "No",      "injury"] = -.5
  X[df_x$injury == "Unknown", "injury"] =   0
  X[df_x$injury == "Yes",     "injury"] =  .5
  X[df_x$family_history_of_seizures == "No",      "history_seizures"] = -.5
  X[df_x$family_history_of_seizures == "Unknown", "history_seizures"] =   0
  X[df_x$family_history_of_seizures == "Yes",     "history_seizures"] =  .5
  
  
  
  return(list("data_list" = data_list,
              "df_time_constant"      = df_x,
              "X"         = X))
  }

