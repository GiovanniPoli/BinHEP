## code to prepare `DATASET` dataset goes here

## G_sim

G_Prior_sim = list(
  "m0" = rep(0, 12),
  "S0" = diag(c(0.1, 1e-04, rep(0.01, 10))),
  "G1" = rbind( cbind( diag(rep(1,2)), matrix(0, nrow = 2, ncol = 10)),
                rep(0, 12),
                cbind( matrix(0, nrow = 9, ncol = 2), diag(rep(1,9)), matrix( 0, ncol = 1, nrow = 9 ))),
  "G2" = rbind( rep(1 / 12, 12), 
                matrix(0, nrow = 11, ncol = 12)),
  "Ve"  = diag(c(1/300, 1/1000, 0.01, rep(1e-04, 9))),
  "Ves" = diag(c(0.1, 1e-04, rep(0.01, 10))),
  "gli" = NULL )

usethis::use_data( G_Prior_sim, overwrite = TRUE)
