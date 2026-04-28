ts_plot = function( binary_ts, eta_ts, Tstar = NULL, 
                    CredInt = NULL, title = NULL,
                    shift = 0.05){
  
  len = length(binary_ts)
  
  df_points = data.frame(x = 1:len, y = NA, col = NA)
  df_points$y[binary_ts == 1] =  1 + shift
  df_points$y[binary_ts == 0] =  0 - shift
  df_points$y[is.na(binary_ts)] = .5
  
  df_points$col[!is.na(binary_ts)] = 1 
  df_points$col[is.na(binary_ts)] = 0 
  
  df_ts = data.frame(y = pnorm(eta_ts), x = 1:len)
  
  plot = ggplot(data = df_points, aes(x = x, y = y, color = as.factor(col))) +
    geom_point() +
    geom_line( data = df_ts,aes(x = x, y = y, color = NULL), col = "blue" )+
    coord_cartesian(ylim = c(0-2*shift,1+2*shift),
                    xlim = c(1, len)) +
    scale_color_manual(NULL,values = c("grey","black"), labels =c("NA","observed")) + 
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(breaks = c(0,0.25,0.5,0.75,1)) +
    geom_hline(yintercept = c(0,1), col = "red", lty = 2) +
    theme_bw() +
    xlab("days") + ylab(TeX("$\\Pr(y=1|\\theta_{jt})$"))+
    ggtitle(NULL)+
    theme(text = element_text(family = "serif"), legend.position = "none")
  if(!is.null(CredInt)){
    CredInt  
    y_poly = pnorm(c(CredInt[1,1],
          CredInt[,1],
          CredInt[len,1],
          CredInt[len,2],
          rev(CredInt[,2]),
          CredInt[1,2]))
    x_poly = c(1,
          1:len,
          len,
          len,
          len:1,
          1)
    df_poly = data.frame(x = x_poly, y = y_poly)
    
    plot = plot+geom_polygon(data = df_poly, 
                   aes(x = x, y = y, color = NULL),
                   fill = "blue",
                   alpha = .1) 
  }
  
  
  if(is.null(Tstar)){
    return(plot)
  }else{
    return(plot + geom_vline( xintercept = c(Tstar), col ="green", lty = 2))
  }
}
getMode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]}
average_sequence_length <- function(row) {
  consecutive_ones <- rle(row)$lengths[rle(row)$values == 1][!is.na(rle(row)$lengths[rle(row)$values == 1])]
  if (length(consecutive_ones) == 0) {
    return(0)
  } else {
    return(mean(consecutive_ones))
  }
}
MyGG_Binary_Heatmap = function( Matrix,
                                Coocorence,
                                cluster,
                                right_space = .2,
                                draw_end_lines = NULL,
                                x_lab = NULL, 
                                y_lab = NULL,
                                gg_title  = NULL,
                                col_label = NULL,
                                row_label = NULL){
  
  
  right_space = right_space * ncol(Matrix)
  xlim = ncol = ncol(Matrix)
  ylim = nrow = nrow(Matrix)
  
  
  xlim = ncol = ncol(Matrix)
  ylim = nrow = nrow(Matrix)
  
  col_order = 1:ncol
  
  RowModel = reorder(as.dendrogram(hclust(dist(Coocorence))), 
                     rowSums(Coocorence))
  
  df.Row   = segment(dendro_data(RowModel, type = "rectangle"))
  
  row_order = order.dendrogram(RowModel)
  
  
  Matrix    = Matrix[row_order, ]
  row_label = row_label[row_order]
  
  # Heatmap
  
  df.heatmap = data.frame("value" = c(Matrix))
  
  df.heatmap$x         = rep(1:ncol, each  = nrow)
  df.heatmap$y         = rep(1:nrow, times = ncol)
  
  rep_col = floor(ncol/nrow)+1
  
  col.2 = c(matrix(cluster[row_order], ncol = rep_col, nrow = nrow) + 1)
  x.2   = c(matrix(c(-(rep_col-1):0) -2, ncol = rep_col, nrow = nrow, byrow = TRUE))
  y.2   = c(matrix(1:nrow, ncol = rep_col, nrow = nrow))
  
  df = rbind(df.heatmap,
             data.frame(value = col.2, x = x.2, y = y.2))
  
  n_cluster = length(unique(cluster))
  
  
  max = max(1,n_cluster)
  
  df$value = factor( x = df$value,
                     levels = c(0,1,  (1:n_cluster) + 1),
                     labels = c("No seizure", "Seizure", 
                                paste0("Partition ", 1:n_cluster)))
  
  col = c("darkblue","yellow", scales::hue_pal()(n_cluster))
  
  plot = ggplot(data = df, aes(x=x, y=y, fill= value)) +
    geom_tile() +
    scale_fill_manual("", values = col , na.value = 'grey') +
    scale_x_continuous(expand = c(0,0), breaks = 1:ncol, labels = col_label) +
    scale_y_continuous(expand = c(0,0), breaks = 1:nrow, labels = row_label) +
    xlab(x_lab)+ylab(y_lab)+ggtitle(gg_title) + theme_minimal() +
    theme(text = element_text(family ="serif"),
          axis.text.y = element_text(),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          panel.grid.major = element_blank(),    #strip major gridlines
          panel.grid.minor = element_blank(),
          legend.position = "none") 
  
  new_x_row  = rep(ncol + 1:right_space, times = nrow)
  new_y_row  = rep(1:nrow, each = right_space)
  
  plot = plot + geom_segment(data = df.Row, aes(x    = y*right_space/max(y) + xlim +.5,       y = x,
                                                xend = yend*right_space/max(yend) + xlim +.5, yend = xend, fill=NULL))
  
  
  
  plot = plot + theme(      panel.grid.major = element_blank(),    #strip major gridlines
                            panel.grid.minor = element_blank(),
                            legend.position = "none")
  
  ### legends 
  df$value = factor( x = df$value,
                     levels = c(0,1,  (1:n_cluster) + 1),
                     labels = c("No seizure", "Seizure", 
                                paste0("Partition ", 1:n_cluster)))
  
  col_siz = col[c(1,2)]
  
  l1 = ggplot(data = data.frame(y = 1,
                                x = 1:3,
                                value = factor( x = c(0,1,NA),
                                                levels = c(0,1),
                                                labels = c("No seizure", "Seizure"))),
              aes(x=x, y=y, fill= value)) +
    geom_tile() +
    scale_fill_manual("", values = col_siz , na.value = 'grey') +
    theme(text = element_text(family ="serif"), legend.position = "top" )
  
  l1 = ggpubr::get_legend(l1)
  
  col_par = col[3:(n_cluster+2)]
  
  l2 = ggplot(data = data.frame(y = 1,
                                x = 1:n_cluster,
                                value = factor( x = 1:n_cluster,
                                                levels =  1:n_cluster,
                                                labels = paste0("Partition ", 1:n_cluster))),
              aes(x=x, y=y, fill= value)) +
    geom_tile() +
    scale_fill_manual("", values = col_par ) +
    theme(text = element_text(family ="serif"), legend.position = "top" )
  
  
  l2 = ggpubr::get_legend(l2)
  
  void_plot = ggplot(data = NULL) + theme_void()
  
  if(!is.null(draw_end_lines)){
    
    end_lines = apply(Matrix,1, function(bv) max(which(!is.na(bv))))
    
    if(draw_end_lines == "A"){
      
      
      df_end_segments = t(sapply(1:length(end_lines),
                                 function(i){
                                   c(
                                     xs = end_lines[i]+0.5,
                                     ys = i+ 0.5,
                                     xend = end_lines[i]+0.5 , 
                                     yend = i- 0.5)
                                 }))
      df_end_segments = data.frame(df_end_segments)
      colnames(df_end_segments) = c("xs","ys","xe","ye")
      
      plot = plot   + geom_segment(data = df_end_segments,
                                   mapping = aes(x    = xs,
                                                 xend = xe,
                                                 y    = ys,
                                                 yend = ye,
                                                 fill = NULL),
                                   linewidth = 1, 
                                   color = "green")
      
    }else{
      
      
      df_lines = data.frame(
        x_lines = rep(end_lines, each = 2) + .5,
        y_lines = rep(1: length(end_lines), each = 2) + 
          rep(c(-.5,+.5), length(end_lines)))  
      
      plot = plot +  geom_path(data = df_lines,
                               mapping = aes(
                                 x = x_lines,
                                 y = y_lines,
                                 fill = NULL), col = "lightblue", linewidth = 1e-6)
      
    }
  }
  
  list(
    "plot" = grid.arrange(
      arrangeGrob(void_plot,l2, void_plot, l1,void_plot,
                  ncol = 5, widths = c(2,10,1,10,2)), plot,
      nrow = 2,
      heights = c(1,10)),
    "legend_1" = l2,
    "legend_2" = l1,
    "heatmap"  = plot)
  
}
My_gg.Traceplot   = function( chain, mean = TRUE, moving_avarage = TRUE,
                                x_label="sample",  y_label = NULL, gg_title = NULL){
  
  alpha = round(length(chain)/100)
  
  plot = ggplot(data = data.frame(chain = chain, x = 1:length(chain)), aes(y = chain, x = x)) + 
    geom_line() + theme_bw() +
    scale_x_continuous(expand = c(0,0)) +
    ylab(y_label) + xlab(x_label) +  ggtitle(gg_title) + 
    theme(text = element_text(family="serif"))
  
  if(mean){
    mean = mean(chain)
    plot = plot + geom_hline( data=data.frame( m = mean),
                              mapping= aes(yintercept = mean), col ="red", lty =4)
  }
  if(moving_avarage){
    
    x_ma = 1:(length(chain)-alpha) + alpha 
    y_ma = sapply(x_ma, function(x) mean(chain[(x-alpha):x]))
    
    plot = plot + geom_line(data = data.frame(x = x_ma, y = y_ma), aes(x = x, y = y), 
                            col = "yellow")
    
  }
  plot
}
My_gg.Heatmap     = function( Matrix, from = NULL, to = NULL, hc_row = TRUE, top_space = 2,
                              hc_col = TRUE, right_space = 2, g_col  = NULL, group_col_palette = NULL,
                              group_row_palette = NULL, group_plot_size_col = 1, g_row  = NULL, 
                              group_plot_size_row = 1, x_lab  = NULL, y_lab = NULL, gg_title=NULL){
  
  if(is.null(from)){from = min(c(Matrix))} 
  if(is.null(to)){to = max(c(Matrix))}
  
  col_label = colnames(Matrix)
  row_label = rownames(Matrix)
  
  xlim = ncol = ncol(Matrix)
  ylim = nrow = nrow(Matrix)
  
  row_order = 1:nrow
  col_order = 1:ncol
  
  # Row
  if(hc_row){
    
    RowModel = reorder(as.dendrogram(hclust(dist(Matrix))), rowSums(Matrix))
    
    df.Row   = segment(dendro_data(RowModel, type = "rectangle"))
    
    row_order = order.dendrogram(RowModel)
    g_row     = g_row[row_order]
    Matrix    = Matrix[row_order, ]
    row_label = row_label[row_order]
    
  }
  
  if(hc_col){
    
    ColModel = reorder(as.dendrogram(hclust(dist(t(Matrix)))), colSums(Matrix))
    
    df.Col   = segment(dendro_data(ColModel, type="rectangle"))
    
    col_order = order.dendrogram(ColModel)
    
    g_col     = g_col[col_order]
    Matrix    = Matrix[ ,col_order]
    col_label = col_label[col_order]
    
  }
  
  # Heatmap
  
  df.heatmap = data.frame("value" = c(Matrix))
  
  df.heatmap$x         = rep(1:ncol, each  = nrow)
  df.heatmap$y         = rep(1:nrow, times = ncol)
  
  plot = ggplot(df.heatmap, aes(x=x, y=y, fill=value)) +
    geom_tile(col="grey") +
    scale_fill_viridis_c("", limits = c(from,to), option = "magma") +
    scale_x_continuous(expand = c(0,0), breaks = 1:ncol, labels = col_label) +
    scale_y_continuous(expand = c(0,0), breaks = 1:nrow, labels = row_label) +
    xlab(x_lab)+ylab(y_lab)+ggtitle(gg_title) + theme_minimal() +
    theme(text = element_text(family ="serif"),
          axis.text.y = element_text(),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
  plot
  
  if(!is.null(g_col)){
    new_y_col  = rep(nrow + 1:group_plot_size_col, times = ncol) 
    new_x_col  = rep(1:ncol, each = group_plot_size_col)
    
    if(is.null(group_col_palette)){group_col_palette = hcl.colors(length(unique(g_col)), palette ="Dark 2")}
    
    
    group_col  = group_col_palette[factor(rep(g_col, each = group_plot_size_col))]
    
    bar_h = ylim
    
    plot = plot + geom_tile(data = data.frame(x = new_x_col, y = new_y_col),
                            mapping = aes(x= x, y=y) , fill = group_col, col="white") + 
      
      geom_segment(data = data.frame(x. = .49, xend. = ncol + .5, y. = nrow + 0.5, yend. = nrow + 0.51),
                   aes(x = x., xend = xend., y = y., yend = yend., fill = NULL), col ="black")
    
    
    plot
    
    ylim = max(new_y_col)
  }  
  
  if(!is.null(g_row)){
    new_x_row  = rep(ncol + 1:group_plot_size_row, times = nrow) 
    new_y_row  = rep(1:nrow, each = group_plot_size_row)
    
    if(is.null(group_row_palette)){group_row_palette = hcl.colors(length(unique(g_row)), palette ="Sunset")}
    
    group_row  = group_row_palette[factor(rep(g_row, each = group_plot_size_row))]
    
    bar_v = xlim
    
    plot = plot + geom_tile(data = data.frame(x = new_x_row, y = new_y_row),
                            mapping = aes(x= x, y=y) , fill = group_row, col="white") +
      geom_segment(data = data.frame(x. = ncol+.5 , xend. = ncol +.5, y. = 0.5, yend. = nrow + .5),
                   aes(x = x., xend = xend., y = y., yend = yend., fill = NULL), col ="black")
    
    xlim = max(new_x_row)
  }  
  
  if(hc_col){
    plot = plot + geom_segment(data = df.Col, aes(x    = x,       y = y/max(y)*top_space    + ylim + .5, 
                                                  xend = xend, yend = yend/max(yend)*top_space + ylim + .5, fill=NULL)) 
  }
  if(hc_row){
    plot = plot + geom_segment(data = df.Row, aes(x    = y*right_space/max(y) + xlim +.5,       y = x,
                                                  xend = yend*right_space/max(yend) + xlim +.5, yend = xend, fill=NULL))
  }
  
  
  plot = plot + theme(      panel.grid.major = element_blank(),    #strip major gridlines
                            panel.grid.minor = element_blank())
  
  plot
  
}
My_gg.Heatmap.cor = function( Matrix, from = NULL, to = NULL, hc_row = TRUE, top_space = 2,
                              hc_col = TRUE, right_space = 2, g_col  = NULL, group_col_palette = NULL,
                              group_row_palette = NULL, group_plot_size_col = 1, g_row  = NULL, 
                              group_plot_size_row = 1, x_lab  = NULL, y_lab = NULL, gg_title=NULL){
  
  if(is.null(from)){from = min(c(Matrix))} 
  if(is.null(to)){to = max(c(Matrix))}
  
  label = colnames(Matrix)
  
  xlim = ncol = ncol(Matrix)
  ylim = nrow = nrow(Matrix)
  
  order = 1:nrow
  
  Model = reorder(as.dendrogram(hclust(dist(Matrix))), rowSums(Matrix))
  
  df.hc   = segment(dendro_data(Model, type = "rectangle"))
  
  order = order.dendrogram(Model)
  
  Matrix    = Matrix[order,order]
  label =  label[order]
  
  df.heatmap = data.frame("value" = c(Matrix))
  
  df.heatmap$x         = rep(1:ncol, each  = nrow)
  df.heatmap$y         = rep(1:nrow, times = ncol)
  
  plot = ggplot(df.heatmap, aes(x=x, y=y, fill=value)) +
    geom_tile(col="grey") +
    scale_fill_viridis_c("", limits = c(from,to)) +
    scale_x_continuous(expand = c(0,0), breaks = 1:ncol, labels = label) +
    scale_y_continuous(expand = c(0,0), breaks = 1:nrow, labels = label) +
    xlab(x_lab)+ylab(y_lab)+ggtitle(gg_title) + theme_minimal() +
    theme(text = element_text(family ="serif"),
          axis.text.y = element_text(),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
  plot
  
  
  
  
  
  
  plot = plot + geom_segment(data = df.hc, aes(x    = x,       y = y/max(y)*top_space    + ylim + .5, 
                                               xend = xend, yend = yend/max(yend)*top_space + ylim + .5, fill=NULL)) 
  
  plot = plot + geom_segment(data = df.hc, aes(x    = y*right_space/max(y) + xlim +.5,       y = x,
                                               xend = yend*right_space/max(yend) + xlim +.5, yend = xend, fill=NULL))
  
  
  plot = plot + theme(      panel.grid.major = element_blank(),    #strip major gridlines
                            panel.grid.minor = element_blank())
  
  plot
  
}
My_gg.Heatmap_group = function( Matrix, from = NULL, to = NULL,  top_space = 2, g_Matrix  = NULL, x_lab  = NULL, y_lab = NULL, gg_title=NULL){
  
  if(is.null(from)){from = min(c(Matrix))} 
  if(is.null(to)){to = max(c(Matrix))}
  
  xlim = ncol = ncol(Matrix)
  ylim = nrow = nrow(Matrix)
  
  Model = reorder(as.dendrogram(hclust(dist(t(Matrix)))), colSums(Matrix))
  
  df.Col   = segment(dendro_data(Model, type="rectangle"))
  
  order = order.dendrogram(Model)
  Matrix   = Matrix[order,order]
  g_Matrix = g_Matrix[,order]
  
  labels =  colnames(Matrix)
  
  # Heatmap
  
  df.heatmap = data.frame("value" = c(Matrix))
  
  df.heatmap$x         = rep(1:ncol, each  = nrow)
  df.heatmap$y         = rep(1:nrow, times = ncol)
  
  plot = ggplot(df.heatmap, aes(x=x, y=y, fill=value)) +
    geom_tile(col="grey") +
    scale_fill_viridis_c("", limits = c(from,to)) +
    scale_x_continuous(expand = c(0,0), breaks = 1:ncol, labels = sapply(labels, TeX)) +
    xlab(x_lab)+ylab(y_lab)+ggtitle(gg_title) + theme_minimal() +
    theme(text = element_text(family ="serif"),
          axis.text.y = element_text(),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
  
  new_y  = rep((nrow + 1):(nrow+nrow(g_Matrix)), times = ncol) 
  new_x  = rep(1:ncol, each = nrow(g_Matrix))
  
  matrix_cols  = matrix(hcl.colors(length(unique(c(g_Matrix))),   palette = "Dark 3")[as.factor(c(g_Matrix))], ncol= ncol)
  
  colr = c(matrix_cols)
  
  bar_h = ylim
  
  ylim = max(new_y)
  
  plot = plot+geom_tile(data = data.frame(x = new_x, y = new_y),
                        mapping = aes(x= x, y=y) , fill = colr, col="white") + 
    scale_y_continuous(expand = c(0,0), breaks = 1:(nrow+nrow(g_Matrix)), labels = sapply(c(labels ,rownames(g_Matrix)),TeX))+
    geom_segment(data = data.frame(x. = .49, xend. = ncol + .5, y. = nrow + 0.5, yend. = nrow + 0.51),
                 aes(x = x., xend = xend., y = y., yend = yend., fill = NULL), col ="black")
  
  #dendogram
  
  plot = plot + geom_segment(data = df.Col, aes(x    = x,       y = y/max(y)*top_space    + ylim + .5, 
                                                xend = xend, yend = yend/max(yend)*top_space + ylim + .5, fill=NULL)) 
  
  
  plot = plot + theme(      panel.grid.major = element_blank(),    #strip major gridlines
                            panel.grid.minor = element_blank())
  
  plot
  
}
