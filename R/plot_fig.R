plot_1 = function(n_simulations = 1e5){
  
  library(ggplot2)
  library(dplyr)
  library(latex2exp)
  library(gridExtra)
  
  set.seed(14)
  
  #### P1 -----
  
  # Parameters
  mu     <- 0
  sigma  <- 1
  lambda <- .5
  
  xi1 <- 0.05
  xi2 <- 0.95
  
  theta      <- seq(-3.5, 3.5, length.out = 2000)
  g          <- dnorm(theta, mean = mu, sd = sigma)
  Phi_lambda <- pnorm(lambda, mean = mu, sd = sigma)
  
  p_xi_g <- ifelse(  theta < lambda,  g / Phi_lambda * xi1, g / (1 - Phi_lambda) * xi2 )
  
  df <- data.frame(
    theta = theta,
    g = g,
    p_xi_g = p_xi_g,
    region = ifelse(theta < lambda, "R_i = 0", "R_i = 1")
  )
  
  df_left  <- df %>% filter(theta < lambda)
  df_right <- df %>% filter(theta >= lambda)
  
  pu = ggplot(df, aes(x = theta)) + 
  geom_area(
    data = df, aes(y = g),
    fill = "#F6C65B", alpha = 0.25)  +
  geom_area(
    data = df, aes(y = p_xi_g),
    fill = "#D7E9F7", alpha = 0.45 ) +
  geom_line( aes(y = g),
      linewidth = 1.2, color = "orange" ) +
  geom_line( aes(y = p_xi_g), 
             linewidth = 1.2, color = "#00A6E8" ) +
  geom_segment( aes(x = min(theta), xend = lambda, y = 0, yend = 0),
                linewidth = 1.2, color = "#00BFC4" ) +
  geom_segment( aes(x = lambda, xend = max(theta), y = 0, yend = 0),
      linewidth = 1.2, color = "#F8766D" ) +
  annotate( "text", x = -1, y = max(df$g), 
      label = "g(theta[i])", parse = TRUE, color = "orange", size = 6, family = "serif") +
  annotate( "text", x = lambda, y = -0.03,
            label = "lambda", parse = TRUE, size = 4, family = "serif") +
  annotate( "text", x = -1.2, y = -0.045,
      label = "R[i] == 0", parse = TRUE, size = 5, family = "serif") +
  annotate("text", x = 1.85, y = -0.045,
      label = "R[i] == 1", parse = TRUE, size = 5, family = "serif") +
  annotate("text", x = -1.6, y = max(df$p_xi_g) * 0.8,
      label = TeX("$P_{\\xi,g} =\\frac{g(\\theta_i) I(\\theta_i < \\lambda)}{\\Phi(\\lambda)}\\xi_{i,1}+\\frac{g(\\theta_i) I(\\theta_i \\geq \\lambda)}{1-\\Phi(\\lambda)}\\xi_{i,2}$"),
      size = 5, color = "navy", family = "serif") +
  coord_cartesian( ylim = c(-.1, max(df$p_xi_g) * 1.05), clip = "off", expand = c(0,0)) +
    theme_classic() + xlab(" ") + ylab("density")+
  ggtitle("A - Example of univariate rescaled model (density).") + 
  theme( text = element_text(family = "serif"),
           legend.position = "none", 
           title = element_text(family = "serif") ) 
  pu
  ## P2 -----
  
  len = 360
  N   = n_simulations
  p   = max(xi1, xi2)
  
  M = matrix(rnorm(len * N, sd = sqrt(1/len)), nrow = N, ncol = len)
  A = t(apply(M, 1, function(x) rnorm(1) + cumsum(x)))
  q = apply(A, 2, function(x) quantile(x, probs = c(0.025,.5,0.975)))
  
  g0 = pnorm(-1.5) * (1 - pnorm(1))
  
  A2 = matrix(NA, nrow = N, ncol = len)
  AC = 0
  
  check  = NULL
  check2 = NULL
  
  while(AC != N){
    
    m0  = rnorm(1)
    th0 = cumsum(rnorm(len, sd = sqrt(1/len)))
    
    ts = m0 + th0
    
    u = runif(1)
    
    M = max(
      p/g0,
      (1-p)/(1-g0)
    )
    
    if( (m0 < -1.5) & (th0[len] > 1) ){
      
      ac = u < (p/g0)/M
      Ri = 1
      
    } else {
      
      ac = u < ((1-p)/(1-g0))/M
      Ri = 2
      
    }
    
    #check = c(check, Ri)
    
    if(ac){
      AC = AC + 1
      A2[AC, ] = ts
      #check2 = c(check2, Ri)
    }
    
    cat("\r", paste0(AC,"/",N), "-", ac, "             ")
  }
  
  
  check2
  
  q2 = apply(A2, 2, function(x) quantile(x, probs = c(0.025,.5,0.975)))
  # lines(q2[1,], type = "l", col = "red")
  # lines(q2[2,], type = "l", col = "red")
  # lines(q2[3,], type = "l", col = "red")
  
  df_p1 = data.frame( x = c( q[1,], rev( q[3,]),
                            q2[1,], rev(q2[3,])),
                      y = c(1:len, len:1),
                      c = as.factor(rep(c(1,2), each = 2*len)))
  df_p2 = data.frame( y = c(q[2,],q2[2,]), 
                      x = c(1:len,1:len), c = as.factor(rep(c(1,2), each = len) ))

  pl = ggplot( ) + 
    geom_polygon( data = df_p1,
      mapping = aes( y = x, x = y, fill = c), alpha = .1, lwd = 0.5) + 
    geom_line( data = df_p2, 
               aes(y = y, x = x, col = c ),lwd = 1) +
    theme_classic() + 
    scale_fill_manual( values = c("orange", "#00A6E8")) +
    scale_color_manual( values = c("orange","#00A6E8")) +
    xlab("time") + ylab(latex2exp::TeX("$E[\\mu_{i}+\\gamma_{i,t}$]")) +
    ggtitle("B - Example of rescaled latent-state (95% C.I.).") +
    coord_cartesian(expand = FALSE, xlim = c(2,358)) +
    theme( text = element_text(family = "serif"),
           legend.position = "none", 
           title = element_text(family = "serif") ) 
  pl
  plot(gridExtra::grid.arrange(pu,pl, ncol = 1))
  ggsave( "New_example_plot_v1.png", pu, width = 19, height = 9, units = "cm", dpi = 1000 )
  ggsave( "New_example_plot_v2.png", pl, width = 19, height = 9, units = "cm", dpi = 1000 )
  ggsave( "New_example_plot.png", gridExtra::grid.arrange(pu,pl, ncol = 1), width = 19, height = 19, units = "cm", dpi = 1000 )
}

