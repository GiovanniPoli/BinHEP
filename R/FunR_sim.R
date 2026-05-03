#' Simulate dataset under Scenario I (balanced groups, 25-25-25-25)
#'
#' @description
#' Generates a synthetic longitudinal binary dataset of \eqn{n = 100}
#' subjects over \eqn{T = 730} days, used in the simulation study of
#' Cantoni, Poli et al.  Subjects are split into four balanced groups of
#' 25: two profiles (\eqn{P_1, P_2}) crossed with two treatment regimes.
#' The four groups have piecewise-constant probit means
#' \eqn{\boldsymbol{\eta}_{p_1}, \ldots, \boldsymbol{\eta}_{p_4}} that
#' change regime at day 366.  Approximately 10\% of the binary
#' observations are set to \code{NA}.  Subjects in
#' \eqn{P_2} mimic a "clamping" behavior — a short time window
#' of length 8–30 days where the success probability is inflated to 0.99.
#'
#' @param seed Integer.  Seed for \code{set.seed()} controlling
#'   reproducibility of the binary draws, the missingness mask, and the
#'   outbreak window placement.
#'
#' @details
#' \strong{Probit means.}  The four group means correspond to two
#' qualitatively different rate dynamics:
#' \itemize{
#'   \item \eqn{P_1} profiles encode \emph{frequent} events that become
#'     rarer in the second period: rates \eqn{1/60} and \eqn{2/60} per
#'     day in the first 365 days (one event every 30–60 days), then
#'     \eqn{2/365} and \eqn{4/365} in the second 365 days.
#'   \item \eqn{G_2} profiles encode \emph{rare} events that become
#'     even rarer: rates \eqn{5/365} and \eqn{10/365} in the first
#'     period, then \eqn{0.5/365} and \eqn{1/365} in the second.
#' }
#'
#' \strong{Group composition.}  25 subjects per profile, four profiles,
#' total \eqn{n = 100}.
#'
#' \strong{Outbreaks.}  All 50 subjects in \eqn{P_2} receive a contiguous
#' window of length 8–30 days, starting uniformly in \eqn{[1, 723]},
#' overwritten with i.i.d. Bernoulli(0.99) draws.
#'
#' \strong{Static covariates.}  Each subject is assigned two factors
#' (\code{f1}, \code{f2}) which are converted to numeric in \eqn{\mathbf{X}}.
#'
#' @return A named list with four fields:
#' \describe{
#'   \item{\code{df_time_constant}}{\eqn{100 \times 2} \code{data.frame}
#'     of subject-level factors \code{f1} and \code{f2}.}
#'   \item{\code{data_list}}{Named list of length 100; element \eqn{i}
#'     contains the time series \code{ts}, treatment vector \code{Trt},
#'     design matrix \code{Z}, and split-point vector \code{Tstar} for
#'     subject \eqn{i}.}
#'   \item{\code{X}}{\eqn{100 \times 3} numeric design matrix with
#'     intercept and centred covariates.}
#'   \item{\code{seed}}{The integer seed passed in (echoed for
#'     traceability).}
#' }
#'
#' @seealso
#'   \code{\link{gen_data_SCENARIO_II}},
#'   \code{\link{gen_data_SCENARIO_III}},
#'   \code{\link{gen_data_SCENARIO_IV}},
#'   \code{where_cut_ts_ext_NA_sim}
#'
#' @examples
#' \dontrun{
#' sim_I <- gen_data_SCENARIO_I(seed = 1)
#' length(sim_I$data_list)        # 100
#' table(sim_I$df_time_constant$f1)
#' }
#'
#' @export
gen_data_SCENARIO_I <- function(seed) {
  
  set.seed(seed)
  
  # Probit means: G1 = frequent events (denom 60), G2 = rare events (denom 365)
  eta_g1 <- qnorm(c(rep(1   / 60,  365), rep(2   / 365, 365)))
  eta_g2 <- qnorm(c(rep(2   / 60,  365), rep(4   / 365, 365)))
  eta_g3 <- qnorm(c(rep(5   / 365, 365), rep(0.5 / 365, 365)))
  eta_g4 <- qnorm(c(rep(10  / 365, 365), rep(1   / 365, 365)))
  
  # 25 subjects per profile
  G1_1 <- t(sapply(1:25, function(x) rbinom(730, 1, pnorm(eta_g1))))
  G1_2 <- t(sapply(1:25, function(x) rbinom(730, 1, pnorm(eta_g2))))
  G2_1 <- t(sapply(1:25, function(x) rbinom(730, 1, pnorm(eta_g3))))
  G2_2 <- t(sapply(1:25, function(x) rbinom(730, 1, pnorm(eta_g4))))
  
  G1 <- rbind(G1_1, G1_2)
  G2 <- rbind(G2_1, G2_2)
  
  # Inject clamping in all G2 subjects (50)
  for (i in 1:50) {
    start_c <- sample.int(723, 1)
    end_c   <- min(730, start_c + sample.int(23, 1) + 7)
    G2[i, start_c:end_c] <- rbinom(end_c - start_c + 1, 1, 99 / 100)
  }
  
  raw_data <- rbind(G1, G2)
  
  # Missingness mask
  raw_data[which(rbinom(730 * 100, 1, 0.1) == 1)] <- NA
  
  raw_list <- lapply(1:100, function(x) {
    s_and_end <- where_cut_ts_ext_NA_sim(raw_data[x, ])
    
    ts  <- raw_data[x, s_and_end[1]:s_and_end[2]]
    Trt <- c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z   <- matrix(1, nrow = length(ts), ncol = 12)
    Z[, 2] <- (c(1:365, 1:365) / 365)[s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1", 2] <- Z[Trt == "Trt.1", 2] + 1 / 365 -
      (1:365 / 365)[s_and_end[1]]
    
    list(
      "ts"    = ts,
      "Trt"   = Trt,
      "Z"     = Z,
      "Tstar" = c(365 + 1 - s_and_end[1], 3650000)
    )
  })
  
  df_time_constant <- data.frame(
    f1 = factor(c(rep(c("Lower risk", "Higher risk",
                        "Lower risk", "Higher risk"), each = 25))),
    f2 = factor(c(rep(c("Level 1", "Level 2"), 50)))
  )
  
  X <- data.matrix(df_time_constant) - 1.5
  X <- cbind(1, X)
  
  names(raw_list) <- rownames(X) <-
    c(paste0(rep("G", 100), rep(c("1", "2"), each = 50), "_", 1:100))
  
  list(
    "df_time_constant" = df_time_constant,
    "data_list"        = raw_list,
    "X"                = X,
    "seed"             = seed
  )
}


#' Simulate dataset under Scenario II (imbalanced 45-45-5-5)
#'
#' @description
#' Same data-generating process as \code{\link{gen_data_SCENARIO_I}} but
#' with strongly imbalanced group sizes: 45 subjects from each
#' \eqn{P_1} profile and only 5 from each \eqn{P_2} profile.  Outbreaks
#' are injected only in the 10 \eqn{P_2} subjects.  This scenario tests
#' the model's behaviour when the "clamping" group is rare.
#'
#' @inheritParams gen_data_SCENARIO_I
#'
#' @details
#' Group composition is 45/45/5/5 instead of 25/25/25/25.  All other
#' elements (probit means, outbreak window length distribution,
#' missingness, covariate construction) are identical to Scenario I.
#' The static covariate \code{f1} is built with \code{times = c(45, 45,
#' 5, 5)} so that its labels match the actual generative groups.
#'
#' @return A named list with the same structure as
#'   \code{\link{gen_data_SCENARIO_I}}.
#'
#' @seealso \code{\link{gen_data_SCENARIO_I}},
#'   \code{\link{gen_data_SCENARIO_III}}, \code{\link{gen_data_SCENARIO_IV}}
#'
#' @export
gen_data_SCENARIO_II <- function(seed) {
  
  set.seed(seed)
  
  eta_g1 <- qnorm(c(rep(1   / 60,  365), rep(2   / 365, 365)))
  eta_g2 <- qnorm(c(rep(2   / 60,  365), rep(4   / 365, 365)))
  eta_g3 <- qnorm(c(rep(5   / 365, 365), rep(0.5 / 365, 365)))
  eta_g4 <- qnorm(c(rep(10  / 365, 365), rep(1   / 365, 365)))
  
  # Imbalanced: 45 G1 + 5 G2 per profile
  G1_1 <- t(sapply(1:45, function(x) rbinom(730, 1, pnorm(eta_g1))))
  G1_2 <- t(sapply(1:45, function(x) rbinom(730, 1, pnorm(eta_g2))))
  G2_1 <- t(sapply(1:5,  function(x) rbinom(730, 1, pnorm(eta_g3))))
  G2_2 <- t(sapply(1:5,  function(x) rbinom(730, 1, pnorm(eta_g4))))
  
  G1 <- rbind(G1_1, G1_2)
  G2 <- rbind(G2_1, G2_2)
  
  # Outbreaks only in 10 G2 subjects
  for (i in 1:10) {
    start_c <- sample.int(723, 1)
    end_c   <- min(730, start_c + sample.int(23, 1) + 7)
    G2[i, start_c:end_c] <- rbinom(end_c - start_c + 1, 1, 99 / 100)
  }
  
  raw_data <- rbind(G1, G2)
  raw_data[which(rbinom(730 * 100, 1, 0.1) == 1)] <- NA
  
  raw_list <- lapply(1:100, function(x) {
    s_and_end <- where_cut_ts_ext_NA_sim(raw_data[x, ])
    
    ts  <- raw_data[x, s_and_end[1]:s_and_end[2]]
    Trt <- c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z   <- matrix(1, nrow = length(ts), ncol = 12)
    Z[, 2] <- (c(1:365, 1:365) / 365)[s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1", 2] <- Z[Trt == "Trt.1", 2] + 1 / 365 -
      (1:365 / 365)[s_and_end[1]]
    
    list(
      "ts"    = ts,
      "Trt"   = Trt,
      "Z"     = Z,
      "Tstar" = c(365 + 1 - s_and_end[1], 3650000)
    )
  })
  
  # FIX: f1 must reflect the actual 45/45/5/5 split, not 25/25/25/25
  df_time_constant <- data.frame(
    f1 = factor(rep(c("Lower risk", "Higher risk",
                      "Lower risk", "Higher risk"),
                    times = c(45, 45, 5, 5))),
    f2 = factor(rep(c("Level 1", "Level 2"), times = 50))
  )
  
  X <- data.matrix(df_time_constant) - 1.5
  X <- cbind(1, X)
  
  names(raw_list) <- rownames(X) <-
    c(paste0(rep("G", 100), rep(c("1", "2"), each = 50), "_", 1:100))
  
  list(
    "df_time_constant" = df_time_constant,
    "data_list"        = raw_list,
    "X"                = X,
    "seed"             = seed
  )
}


#' Simulate dataset under Scenario III (imbalanced 5-5-45-45)
#'
#' @description
#' Mirror image of \code{\link{gen_data_SCENARIO_II}}: group composition
#' is 5/5/45/45 — the \eqn{P_2} ("clamping") profiles are now the
#' majority.
#'
#' @inheritParams gen_data_SCENARIO_I
#'
#' @details
#' Group composition is 5/5/45/45.  All other elements are identical to
#' Scenarios I–II.  The static covariate \code{f1} is built with
#' \code{times = c(5, 5, 45, 45)} so that its labels match the actual
#' generative groups.
#'
#' @return A named list with the same structure as
#'   \code{\link{gen_data_SCENARIO_I}}.
#'
#' @seealso \code{\link{gen_data_SCENARIO_I}},
#'   \code{\link{gen_data_SCENARIO_II}}, \code{\link{gen_data_SCENARIO_IV}}
#'
#' @export
gen_data_SCENARIO_III <- function(seed) {
  
  set.seed(seed)
  
  eta_g1 <- qnorm(c(rep(1   / 60,  365), rep(2   / 365, 365)))
  eta_g2 <- qnorm(c(rep(2   / 60,  365), rep(4   / 365, 365)))
  eta_g3 <- qnorm(c(rep(5   / 365, 365), rep(0.5 / 365, 365)))
  eta_g4 <- qnorm(c(rep(10  / 365, 365), rep(1   / 365, 365)))
  
  # Imbalanced: 5 G1 + 45 G2 per profile
  G1_1 <- t(sapply(1:5,  function(x) rbinom(730, 1, pnorm(eta_g1))))
  G1_2 <- t(sapply(1:5,  function(x) rbinom(730, 1, pnorm(eta_g2))))
  G2_1 <- t(sapply(1:45, function(x) rbinom(730, 1, pnorm(eta_g3))))
  G2_2 <- t(sapply(1:45, function(x) rbinom(730, 1, pnorm(eta_g4))))
  
  G1 <- rbind(G1_1, G1_2)
  G2 <- rbind(G2_1, G2_2)
  
  # Outbreaks in 90 G2 subjects
  for (i in 1:90) {
    start_c <- sample.int(723, 1)
    end_c   <- min(730, start_c + sample.int(23, 1) + 7)
    G2[i, start_c:end_c] <- rbinom(end_c - start_c + 1, 1, 99 / 100)
  }
  
  raw_data <- rbind(G1, G2)
  raw_data[which(rbinom(730 * 100, 1, 0.1) == 1)] <- NA
  
  raw_list <- lapply(1:100, function(x) {
    s_and_end <- where_cut_ts_ext_NA_sim(raw_data[x, ])
    
    ts  <- raw_data[x, s_and_end[1]:s_and_end[2]]
    Trt <- c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z   <- matrix(1, nrow = length(ts), ncol = 12)
    Z[, 2] <- (c(1:365, 1:365) / 365)[s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1", 2] <- Z[Trt == "Trt.1", 2] + 1 / 365 -
      (1:365 / 365)[s_and_end[1]]   # correction
    
    list(
      "ts"    = ts,
      "Trt"   = Trt,
      "Z"     = Z,
      "Tstar" = c(365 + 1 - s_and_end[1], 3650000)
    )
  })
  
  # FIX: f1 must reflect the actual 5/5/45/45 split, not 25/25/25/25
  df_time_constant <- data.frame(
    f1 = factor(rep(c("Lower risk", "Higher risk",
                      "Lower risk", "Higher risk"),
                    times = c(5, 5, 45, 45))),
    f2 = factor(rep(c("Level 1", "Level 2"), times = 50))
  )
  
  X <- data.matrix(df_time_constant) - 1.5
  X <- cbind(1, X)
  
  names(raw_list) <- rownames(X) <-
    c(paste0(rep("G", 100), rep(c("1", "2"), each = 50), "_", 1:100))
  
  list(
    "df_time_constant" = df_time_constant,
    "data_list"        = raw_list,
    "X"                = X,
    "seed"             = seed
  )
}


#' Simulate dataset under Scenario IV (model-based DGP)
#'
#' @description
#' Unlike Scenarios I–III, which use ad-hoc piecewise-constant probit
#' means, Scenario IV draws the binary time series directly from the
#' DLM-probit prior \eqn{g} via
#' \code{\link{sample_biniary_ts_wSplit_from_G}}.  This scenario tests
#' model recovery under against a correctly-specified two-step approach, where after a 
#' joint static probit model, data are generated from the same state-space model used for inference.
#'
#' Two intercept levels (\eqn{\mu_{p_1} = -3}, \eqn{\mu_{p_2} = -2.5})
#' are applied to the first 50 and last 50 subjects respectively, all
#' sharing the same DLM parameters.
#'
#' @param seed  Integer seed for \code{set.seed()}.
#' @param G_par Named list of state-space hyperparameters passed to
#'   \code{\link{sample_biniary_ts_wSplit_from_G}}:
#'   \describe{
#'     \item{\code{m0}, \code{S0}}{Initial state mean and covariance.}
#'     \item{\code{G1}, \code{G2}}{Baseline and alternative-regime
#'       transition matrices.}
#'     \item{\code{Ve}, \code{Ves}}{Innovation covariances for the two
#'       regimes.}
#'   }
#'
#' @details
#' Subjects are simulated over \eqn{T = 730} days with a regime switch
#' at day 366 (\code{Tstar = c(366, 3650000)}).  After generation, 10\%
#' of the binary observations are masked to \code{NA}.
#'
#' @return A named list with the same structure as
#'   \code{\link{gen_data_SCENARIO_I}}.
#'
#' @seealso
#'   \code{\link{sample_biniary_ts_wSplit_from_G}},
#'   \code{\link{gen_data_SCENARIO_I}}
#'
#' @export
gen_data_SCENARIO_IV <- function(seed, G_par) {
  
  set.seed(seed)
  
  mu_g1 <- -3
  mu_g2 <- -2.5
  
  raw_data   <- matrix(NA, ncol = 730, nrow = 100)
  Z_all      <- matrix(1, nrow = 730, ncol = 12)
  Z_all[, 2] <- (c(1:365, 1:365) / 365)
  
  for (i in 1:50) {
    raw_data[i, ] <- sample_biniary_ts_wSplit_from_G(
      mu_g1, TT = 730, matFF = Z_all,
      mu0 = G_par$m0, Sigma0 = G_par$S0,
      G = G_par$G1, G_star = G_par$G2,
      SigmaEps = G_par$Ve, SigmaEps_star = G_par$Ves,
      Tstar = c(366, 3650000))  
  }
  for (i in 51:100) {
    raw_data[i, ] <- sample_biniary_ts_wSplit_from_G(
      mu_g2, TT = 730, matFF = Z_all,
      mu0 = G_par$m0, Sigma0 = G_par$S0,
      G = G_par$G1, G_star = G_par$G2,
      SigmaEps = G_par$Ve, SigmaEps_star = G_par$Ves,
      Tstar = c(366, 3650000))  
  }
  
  raw_data[which(rbinom(730 * 100, 1, 0.1) == 1)] <- NA
  
  raw_list <- lapply(1:100, function(x) {
    s_and_end <- where_cut_ts_ext_NA_sim(raw_data[x, ])
    
    ts  <- raw_data[x, s_and_end[1]:s_and_end[2]]
    Trt <- c(rep("Trt.1", 365), rep("Trt.2", 365))[s_and_end[1]:s_and_end[2]]
    Z   <- matrix(1, nrow = length(ts), ncol = 12)
    Z[, 2] <- (c(1:365, 1:365) / 365)[s_and_end[1]:s_and_end[2]]
    Z[Trt == "Trt.1", 2] <- Z[Trt == "Trt.1", 2] + 1 / 365 -
      (1:365 / 365)[s_and_end[1]]
    
    list(
      "ts"    = ts,
      "Trt"   = Trt,
      "Z"     = Z,
      "Tstar" = c(365 + 1 - s_and_end[1], 3650000)
    )
  })
  
  df_time_constant <- data.frame(
    f1 = factor(c(rep(c("Lower risk", "Higher risk",
                        "Lower risk", "Higher risk"), each = 25))),
    f2 = factor(c(rep(c("Level 1", "Level 2"), 50)))
  )
  
  X <- data.matrix(df_time_constant) - 1.5
  X <- cbind(1, X)
  
  names(raw_list) <- rownames(X) <-
    c(paste0(rep("G", 100), rep(c("1", "2"), each = 50), "_", 1:100))
  
  list(
    "df_time_constant" = df_time_constant,
    "data_list"        = raw_list,
    "X"                = X,
    "seed"             = seed
  )
}