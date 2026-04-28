#include <iostream>
#include <chrono>
#include <thread>
#include <RcppArmadillo.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

// Internal: generate the integer index sequence 0, 1, ..., n-1.
arma::uvec rcpp_index_gen(const int n) {
  arma::uvec ret(n);
  for(int i = 0; i < n; ++i) {
    ret(i) = i ;    
  }
  return(ret);
}

// Internal: rbind
arma::mat cpp_rbind(const arma::mat& mat1, const arma::mat& mat2) {
  arma::mat result(mat1.n_rows + mat2.n_rows, mat1.n_cols);
  result.rows(0, mat1.n_rows - 1) = mat1;
  result.rows(mat1.n_rows, mat1.n_rows + mat2.n_rows - 1) = mat2;
  return result;
}

// Internal: pop 1 elem from an arma::ivec
arma::ivec rcpp_remove_element(const arma::ivec & vector, int position) {
  
  int old_ln = vector.n_elem ; 
  arma::ivec ret( old_ln - 1);
  
  if(position == 0){
    ret = vector.subvec(1, old_ln - 1);
  }else if(position == (old_ln - 1) ){
    ret = vector.subvec(0, old_ln - 2);
  }else{
    ret.subvec(0, position - 1) = vector.subvec(0, position - 1);
    ret.subvec(position, old_ln - 2) = vector.subvec(position + 1, old_ln - 1);
  } 
  return ret;
} 

// Internal: remove one row from arma::mat
arma::mat rcpp_remove_row(const arma::mat & matrix, int row) {
  
  int old_n_rows = matrix.n_rows ;  
  int n_cols     = matrix.n_cols ;
  
  arma::mat ret( old_n_rows - 1, n_cols ) ;
  
  
  if(row == 0){
    ret = matrix.submat(1,0,old_n_rows-1,n_cols-1);
  }else if(row == (old_n_rows - 1) ){  
    ret = matrix.submat(0,0,old_n_rows-2,n_cols-1);
  }else{ 
    ret.submat(0,0,row-1,n_cols-1) = matrix.submat(0,0,row-1,n_cols-1);
    ret.submat(row,0,old_n_rows-2,n_cols-1) = matrix.submat(row+1,0,old_n_rows-1,n_cols-1);
  } 
  return ret;
} 

// Internal: table
arma::ivec rcpp_arma_table( const arma::vec & x) {
  std::map<int, int> counts;
  int n = x.n_elem ;
  for (int i = 0; i < n; i++) {
    counts[x(i)]++;
  } 
  arma::ivec vec(counts.size());
  int index = 0;
  for (const auto& pair : counts) {
    vec(index++) = pair.second;
  } 
  return vec;
}  

// Internal: add one singleton
void rcpp_add_elem_1(arma::ivec& v) {
  v.resize(v.n_rows + 1);  
  v.row(v.n_rows - 1) = 1; 
}  

// Internal: add Xis vector (row) to the matrix of the parameters for new cluster
void rcpp_add_row(arma::mat& m, const arma::vec row) {
  m.resize(m.n_rows + 1, m.n_cols); // Aumenta la dimensione di una riga
  m.row(m.n_rows - 1) = row.t();    // Aggiunge la riga alla fine
}  

// Re-labeling
arma::ivec rcpp_correct_labels( arma::ivec x, int target) {
  int n = x.n_elem ;
  for (int i = 0; i < n; i++) {
    if( x(i) > target){
      x(i) -= 1 ;
    }
  }  
  return x;
} 


// internal rmvnorm
arma::mat cpp_mvrnormArma(int n, const arma::vec & mu,const  arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

arma::mat cpp_mvrnormArma_transposed(int n, const arma::vec & mu, const arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(ncols, n);
  return arma::repmat(mu, 1, n) + arma::chol(sigma).t() * Y;
}

arma::mat cpp_mvrnormArma_spectral(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(ncols, n);

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, sigma);
  arma::mat transform = eigvec * arma::diagmat(arma::sqrt(eigval)) * eigvec.t();

  return arma::repmat(mu, 1, n).t() + Y.t() * transform;
}

arma::mat cpp_mvrnormArma_transposed_spectral(int n, const arma::vec & mu,
                                                     const arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(ncols, n);

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, sigma);
  arma::mat transform = eigvec * arma::diagmat(arma::sqrt(eigval)) * eigvec.t();

  return arma::repmat(mu, 1, n) + transform.t() * Y;
}

arma::colvec cpp_mvrnormArma1(const arma::vec & mu, const arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::colvec Z = arma::randn(ncols);
  return mu + arma::chol(sigma).t() * Z;
}


//' @name mvrnormArma
//' @title Multivariate normal sampler (Cholesky, user-facing)
//'
//' @description
//' Wrappers around the internal Cholesky-based multivariate normal 
//' sampler.  \code{mvrnormArma} returns \eqn{n} samples as an \eqn{n \times p}
//' matrix; \code{mvrnormArma1} returns a single draw as a column vector.
//'
//' The covariance matrix \eqn{\boldsymbol{\Sigma}} must be symmetric and
//' positive definite.  Armadillo's \code{chol()} will throw a runtime error if
//' the matrix is not positive definite.
//'
//' @param n     (\code{mvrnormArma} only) Number of samples to draw.
//' @param mu    Mean vector of length \eqn{p}.
//' @param sigma Symmetric positive-definite covariance matrix (\eqn{p \times p}).
//'
//' @return
//' \describe{
//'   \item{\code{mvrnormArma}}{An \eqn{n \times p} numeric matrix; rows are
//'     independent draws from \eqn{\mathcal{N}_p(\boldsymbol{\mu},
//'     \boldsymbol{\Sigma})}.}
//'   \item{\code{mvrnormArma1}}{A numeric column vector of length \eqn{p}.}
//' }
//'
//' @examples
//' Sigma <- diag(c(1, 2))
//' mu    <- c(0, 0)
//'
//' # Draw 100 samples
//' samp <- mvrnormArma(n = 100, mu = mu, sigma = Sigma)
//' colMeans(samp)          # should be close to mu
//' cov(samp)               # should be close to Sigma
//'
//' # Single draw
//' mvrnormArma1(mu = mu, sigma = Sigma)
//'
//' @seealso \code{\link{cpp_mvrnormArma}}, \code{\link{cpp_mvrnormArma1}},
//'   \code{\link{cpp_mvrnormArma_spectral}}
// [[Rcpp::export]]
arma::mat mvrnormArma(int n, const arma::vec & mu, const arma::mat & sigma) {
  return cpp_mvrnormArma(n, mu, sigma);
}

//' @rdname mvrnormArma
// [[Rcpp::export]]
arma::colvec mvrnormArma1(const arma::vec & mu, const arma::mat & sigma) {
  return cpp_mvrnormArma1(mu, sigma);
}


// ============================================================
//  Truncated normal samplers
// ============================================================

//' Vectorised quantile function of the truncated normal distribution
//'
//' Based on the univariate function of \code{RcppDist}.
//' Evaluates the inverse CDF of the truncated normal
//' distribution element-wise over vectors of probabilities, means, and
//' truncation bounds.  Each element \eqn{i} of the output is
//' \eqn{Q_{\text{TN}}(p_i;\, \mu_i,\, 1,\, a_i,\, b_i)}, i.e. the quantile
//' at probability \eqn{p_i} of a \eqn{\mathcal{N}(\mu_i, 1)} distribution
//' truncated to \eqn{[a_i, b_i]}.
//'
//' Internally delegates to \code{q_truncnorm()} from the \pkg{RcppDist}
//' \code{truncnorm} header, which uses the closed-form expression based on the
//' standard normal CDF.
//'
//' @param p  Vector of probabilities in \eqn{(0, 1)}.
//' @param mu Vector of location parameters (means before truncation).
//' @param a  Vector of lower truncation bounds (\code{-Inf} is allowed).
//' @param b  Vector of upper truncation bounds (\code{Inf} is allowed).
//'
//' @details
//' All input vectors must have the same length.  The standard deviation is
//' fixed at 1; scale \code{mu}, \code{a}, and \code{b} accordingly if a
//' different scale is needed.
//'
//' @return Numeric vector of quantiles of the same length as \code{p}.
//'
//' @seealso \code{\link{rcpp_rtnorm_vec}}
//'
//' @examples
//' # 50th percentile of N(0,1) truncated to [0, Inf)
//' rcpp_qtnorm_vec(p = 0.5, mu = 0, a = 0, b = Inf)
// [[Rcpp::export]]
arma::vec rcpp_qtnorm_vec( const arma::vec & p,
                           const arma::vec & mu,
                           const arma::vec & a,
                           const arma::vec & b) {
  arma::vec sample(p.n_elem);
  for (int i = 0; i < (int)p.n_elem; ++i)
    sample(i) = q_truncnorm(p(i), mu(i), 1.0, a(i), b(i));
  return sample;
}


//' Vectorised random sampler from the truncated normal distribution
//' 
//' Based on the univariate function of \code{RcppDist}.
//' Draws one independent variate per element from a (possibly distinct)
//' truncated normal distribution.  Each draw \eqn{i} is a realisation of
//' \eqn{\mathcal{N}(\mu_i, 1)} truncated to \eqn{[a_i, b_i]}.
//'
//' Internally delegates to \code{r_truncnorm()} from the \pkg{RcppDist}
//' \code{truncnorm} header.
//'
//' @param mu Vector of location parameters (means before truncation).
//' @param a  Vector of lower truncation bounds (\code{-Inf} is allowed).
//' @param b  Vector of upper truncation bounds (\code{Inf} is allowed).
//'
//' @details
//' The standard deviation is fixed at 1.  All vectors must have the same
//' length.  This function is especially suited to block-updating steps in
//' Gibbs samplers for probit or tobit models, where each latent variable has
//' its own mean and truncation region.
//'
//' @return Numeric vector of random draws of the same length as \code{mu}.
//'
//' @seealso \code{\link{rcpp_qtnorm_vec}}
//'
//' @examples
//' # Draw from N(0,1) truncated to [0, Inf) and N(1,1) truncated to [-1, 1]
//' rcpp_rtnorm_vec(mu = c(0, 1), a = c(0, -1), b = c(Inf, 1))
// [[Rcpp::export]]
arma::vec rcpp_rtnorm_vec(const arma::vec & mu,
                          const arma::vec & a,
                          const arma::vec & b) {
  arma::vec sample(mu.n_elem);
  for (int i = 0; i < (int)mu.n_elem; ++i)
    sample(i) = r_truncnorm(mu(i), 1.0, a(i), b(i));
  return sample;
}


// ============================================================
//  Dirichlet samplers
// ============================================================

//' Draw a single sample from a Dirichlet distribution
//'
//' Generates one draw from \eqn{\text{Dir}(\boldsymbol{\alpha})} using the
//' standard Gamma-ratio method: independently sample
//' \eqn{x_k \sim \text{Gamma}(\alpha_k, 1)} for \eqn{k = 1, \ldots, K} and
//' return \eqn{\mathbf{x} / \sum_k x_k}.
//'
//' @param parameters Concentration parameter vector
//'   \eqn{\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)^\top} with all
//'   elements strictly positive.
//'
//' @return Numeric vector of length \eqn{K} summing to 1, representing a
//'   probability simplex point.
//'
//' @seealso \code{\link{cpp_rdirichletArma1_plus_1_pos}}
//'
//' @examples
//' # Symmetric Dirichlet with concentration 1 (equivalent to Uniform on simplex)
//' cpp_rdirichletArma1(parameters = c(1, 1, 1))
//'
//' # Informative prior
//' cpp_rdirichletArma1(parameters = c(10, 2, 5))
// [[Rcpp::export]]
arma::vec cpp_rdirichletArma1(const arma::colvec & parameters) {
  int p = parameters.n_elem;
  arma::vec ret(p);
  for (int i = 0; i < p; i++)
    ret(i) = arma::randg(1, arma::distr_param(parameters(i), 1.0))(0);
  ret = ret / arma::sum(ret);
  return ret;
}

arma::vec cpp_rdirichletArma1_plus_1_pos(const arma::colvec & parameters, int pos) {
  int p = parameters.n_elem;
  arma::vec ret(p);
  for (int i = 0; i < p; i++) {
    double alpha = (i == pos) ? parameters(i) + 1.0 : parameters(i);
    ret(i) = arma::randg(1, arma::distr_param(alpha, 1.0))(0);
  }
  ret = ret / arma::sum(ret);
  return ret;
}

// ============================================================
//  Weighted sampling
// ============================================================

//' Weighted sampling with replacement from an integer vector
//'
//' Draws \code{size} indices from \code{vec} with replacement, where the
//' probability of drawing element \eqn{i} is proportional to
//' \code{prob[i]}.  Delegates to
//' \code{RcppArmadillo::sample()}, which mirrors R's \code{sample()} but
//' operates directly on \code{arma::uvec} objects, avoiding unnecessary
//' copies.
//'
//' @param vec  Non-negative integer vector of candidates to sample from.
//' @param size Number of draws (positive integer).
//' @param prob Numeric vector of non-negative sampling weights of the same
//'   length as \code{vec}.  Need not sum to 1; normalisation is handled
//'   internally.
//'
//' @return An \code{arma::uvec} (unsigned integer vector) of length
//'   \code{size} containing the sampled elements.
//'
//' @examples
//' candidates <- 0:4
//' weights    <- c(0.1, 0.4, 0.2, 0.2, 0.1)
//' rcpp_sample(vec = candidates, size = 10L, prob = weights)
// [[Rcpp::export]]
arma::uvec rcpp_sample(const arma::uvec vec,
                        const int size,
                        const arma::colvec prob) {
  return RcppArmadillo::sample(vec, size, true, prob);
}


//' Draw a trajectory from the state-space posterior via a Sequential Monte Carlo sampler 
//'
//' @description
//' Implements an particle filter for a binary, probit-linked Dynamic Model (DLM) with two transition
//' regimes. After the forward pass, a single trajectory
//' \eqn{\{\mathbf{x}_t\}_{t=1}^{T}} is extracted by backward tracing through
//' the stored indices.
//'
//' @param y
//'   Integer vector of length \eqn{T} containing the binary observations
//'   (0 or 1); \code{NA} entries are treated as missing.
//' @param delta
//'   Numeric vector of length \eqn{q} of fixed regression coefficients for
//'   the static covariate \code{x}, defining the scalar offset
//'   \eqn{\mu^* = \mathbf{x}^\top \boldsymbol{\delta}}.
//' @param x
//'   Numeric vector of length \eqn{q} of static covariates entering the probit
//'   mean through \eqn{\mu^*}.
//' @param matFF
//'   \eqn{T \times p} observation design matrix.
//' @param mu0
//'   Numeric vector of length \eqn{p}: prior mean of the initial state
//'   \eqn{\mathbf{x}_0 \sim \mathcal{N}_p(\boldsymbol{\mu}_0,
//'   \boldsymbol{\Sigma}_0)}.
//' @param Sigma0
//'   \eqn{p \times p} symmetric positive-definite prior covariance of the
//'   initial state.
//' @param G
//'   \eqn{p \times p} state transition matrix for the standard regime,
//'   \eqn{\mathbf{x}_t = \mathbf{G}\,\mathbf{x}_{t-1} +
//'   \boldsymbol{\varepsilon}_t}, \eqn{\boldsymbol{\varepsilon}_t \sim
//'   \mathcal{N}_p(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)}.
//' @param G_star
//'   \eqn{p \times p} state transition matrix for the alternative regime,
//'   applied at the time points listed in \code{Tstar}.
//' @param SigmaEps
//'   \eqn{p \times p} state evolution covariance for the baseline regime.
//' @param SigmaEps_star
//'   \eqn{p \times p} state evolution covariance for the alternative regime.
//' @param Tstar
//'   Integer vector of time indices at which the alternative regime
//'   transition (\code{G_star}, \code{SigmaEps_star}) is used instead of the
//'   baseline.
//' @param nSim
//'   Number of SMC particles.  Larger values reduce Monte Carlo variance at
//'   the cost of memory and time per step.
//'   Default: \code{500}.
//'
//' @return
//' A \eqn{T \times p} numeric matrix.  Row \eqn{t} contains the state vector
//' \eqn{\mathbf{x}_t} of the single trajectory drawn from
//' \eqn{p(\mathbf{x}_{1:T} \mid y_{1:T})}.
//'
//' @seealso
//' \code{\link{cpp_mvrnormArma_transposed}}, \code{\link{rcpp_rtnorm_vec}},
//' \code{\link{rcpp_sample}}, \code{\link{rcpp_index_gen}}
//'
//' @examples
//' \dontrun{
//' set.seed(14041995)
//' TT  <- 50
//' p   <- 2
//' y   <- rbinom(TT, 1, 0.6)
//' y[sample(TT, 5)] <- NA          # inject some missing observations
//'
//' matFF   <- matrix(c(rep(1, TT), seq_len(TT)), nrow = TT)
//' mu0     <- rep(0, p)
//' Sigma0  <- diag(p)
//' G       <- diag(p) * 0.05
//' G_star  <- diag(p) * 0.80
//' SigE    <- diag(p) * 0.1
//' SigE_s  <- diag(p) * 1
//' delta   <- c(0.1, -0.1)
//' x       <- c(1, 0)
//' Tstar   <- c(10L, 30L)          # regime switches at t = 10 and t = 30
//'
//' traj <- rcpp_rTheta_conditioned_RaoBlackwellAuxSMC(
//'   y          = as.integer(y),
//'   delta      = delta,
//'   x          = x,
//'   matFF      = matFF,
//'   mu0        = mu0,
//'   Sigma0     = Sigma0,
//'   G          = G,
//'   G_star     = G_star,
//'   SigmaEps   = SigE,
//'   SigmaEps_star = SigE_s,
//'   Tstar      = Tstar,
//'   nSim       = 500L
//' )
//' dim(traj)    # TT x p
//' }
//' 
// [[Rcpp::export]]
arma::mat rcpp_rTheta_conditioned_RaoBlackwellAuxSMC( 
     const Rcpp::IntegerVector & y,
     const arma::colvec & delta,
     const arma::colvec & x,
     const arma::mat & matFF,
     const arma::colvec & mu0,
     const arma::mat & Sigma0,
     const arma::mat & G,
     const arma::mat & G_star, 
     const arma::mat & SigmaEps,
     const arma::mat & SigmaEps_star,
     const Rcpp::IntegerVector & Tstar,
     const int nSim = 500) {
   
   // Const.
   const int         p  = mu0.n_elem ;
   const int         TT = y.length() ;
   const arma::uvec  x_order = rcpp_index_gen(nSim) ;
   
   const arma::mat    Gt = G.t() ;
   const arma::mat    Gt_star = G_star.t() ;
   const double       mu_star = arma::as_scalar( x.t() * delta ) ;
   const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
   
   // Init. the MC sample 
   arma::cube SMCret( TT, p, nSim );
   
   // SMC stuffs 
   int star_count = 0; 
   int star_times_minus_1 = Tstar.length() - 1;
   
   arma::mat P_F(p,p) ;
   
   arma::colvec FF(p);
   arma::rowvec FFt(p);
   arma::mat    FFtFF(p, p);
   
   arma::mat    P_P(p, p);
   arma::colvec yast_P(nSim);
   arma::colvec log_ws_IS_it(nSim);
   arma::colvec yast_(nSim);
   double S_P;
   
   arma::colvec xhat_col_vec(nSim);
   
   
   arma::uvec order_rss(nSim);
   arma::umat order_matrix(TT, nSim, arma::fill::value(TT));
   
   arma::cube subCube ;
   
   arma::mat xPart   = cpp_mvrnormArma_transposed( nSim, mu0, Sigma0) ;
   arma::mat xhat_F = arma::repmat(mu0, 1, nSim);
   arma::mat xhat_P(p, nSim);
   
   for(int tt = 0; tt < TT; ++tt) {
     
     FFt   = matFF.submat(tt, 0, tt, p-1) ;
     FF    = FFt.t();
     FFtFF = FF * FFt;
     
     if( y(tt) == NA_INTEGER){
       // Prior Evaluate evolutions
       if(tt != (Tstar(star_count)-1)){
         xhat_P  = G * xPart ;
         xPart   = xhat_P + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, SigmaEps) ;
       }else{
         xhat_P  = G_star * xPart ;
         xPart   = xhat_P + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, SigmaEps_star) ;
         if(star_times_minus_1 != star_count) star_count = star_count + 1;
       }  
     }else{
       
       // Evaluate Evolutions
       if(tt != (Tstar(star_count)-1)){
         xhat_P  = G * xPart;                      
         P_P     = SigmaEps ;  
       }else{  
         xhat_P  = G_star * xPart;                      
         P_P     = SigmaEps_star ;
         if(star_times_minus_1 != star_count) star_count = star_count + 1;
       }  
       
       yast_P  = (FFt * xhat_P).t();
       S_P     = arma::as_scalar(FFt*P_P*FF + 1.0);
       
       // Compute weights
       log_ws_IS_it = arma::log(
         arma::normcdf(  (2.0 * y(tt) - 1.0) * (yast_P + mu_star )
                           / std::sqrt(S_P) ) ) ; 
       
       log_ws_IS_it = log_ws_IS_it - arma::max(log_ws_IS_it);
       log_ws_IS_it.replace(arma::datum::nan, -100.0) ;
       
       // Resample step (y_{it} = 1)
       order_rss = rcpp_sample(x_order, nSim, arma::exp(log_ws_IS_it) );
       order_matrix.row(tt) = order_rss.t()  ;
       
       yast_P    = yast_P(order_rss);
       xhat_P    = xhat_P.cols(order_rss);
       
       
       switch( y(tt) ) {
       case 1 :
         for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
           yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + mu_star, 
                 std::sqrt(S_P), 0, R_PosInf);
         }
         break;
       case 0 :
         for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
           yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + mu_star, 
                 std::sqrt(S_P), R_NegInf, 0);
         }
         break;
       default:
         Rcpp::Rcout << "Warnings: the ts is not binary" << std::endl;
       }
       
       // MC Evolution
       xhat_col_vec = yast_ - yast_P - mu_star   ;
       xhat_F       = xhat_P + P_P * FF * xhat_col_vec.t() / S_P ;
       P_F          = P_P - P_P * FFtFF * P_P / S_P;
       xPart        = xhat_F + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, P_F) ;
     }
     SMCret.row(tt) = xPart ;
   } 
   
   int old_pos = 0 ;
   arma::mat RetMat(TT,p) ;
   for(int tt = 0; tt < TT-1 ; ++tt){
     RetMat.row( TT-tt-1) = SMCret.slice( old_pos ).row( TT-tt-1 ) ;
     if( order_matrix(TT-tt-1, old_pos) != TT ){
       old_pos = order_matrix(TT-tt-1, old_pos) ;
     }
   }
   RetMat.row(0) = SMCret.slice(old_pos).row(0) ;
   return RetMat;
}

//' Sample the centered linear predictor trajectory from a Rao-Blackwellised
//' auxiliary SMC
//'
//' @description
//' Runs the full forward auxiliary particle filter and backward trajectory
//' extraction of
//' \code{\link{rcpp_rTheta_conditioned_RaoBlackwellAuxSMC}}, but instead of
//' returning the latent state matrix, it returns the a centered linear predictor as
//' a vector. 
//'
//' @inheritParams rcpp_rTheta_conditioned_RaoBlackwellAuxSMC
//'
//' @return A numeric vector of length \eqn{T}. 
//'
//' @seealso \code{\link{rcpp_rTheta_conditioned_RaoBlackwellAuxSMC}}
//'
//' @examples
//' \dontrun{
//' eta <- rcpp_rCenteredLinearPredictor_RaoBlackwellAuxSMC(
//'   y             = as.integer(y),
//'   delta         = delta,
//'   x             = x_cov,
//'   matFF         = matFF,
//'   mu0           = mu0,
//'   Sigma0        = Sigma0,
//'   G             = G,
//'   G_star        = G_star,
//'   SigmaEps      = SigE,
//'   SigmaEps_star = SigE_s,
//'   Tstar         = Tstar,
//'   nSim          = 500L
//' )
//' length(eta)   # T
//' # Full probit mean at each t: eta + mu_star
//' }
//' 
// [[Rcpp::export]]
arma::vec rcpp_rEta_conditioned_RaoBlackwellAuxSMC( 
    const Rcpp::IntegerVector & y,
    const arma::colvec & delta,
    const arma::colvec & x,
    const arma::mat & matFF,
    const arma::colvec & mu0,
    const arma::mat & Sigma0,
    const arma::mat & G,
    const arma::mat & G_star, 
    const arma::mat & SigmaEps,
    const arma::mat & SigmaEps_star,
    const Rcpp::IntegerVector & Tstar,
    const int nSim = 500) {
  
  // Const.
  const int         p  = mu0.n_elem ;
  const int         TT = y.length() ;
  const arma::uvec  x_order = rcpp_index_gen(nSim) ;
  
  const arma::mat    Gt = G.t() ;
  const arma::mat    Gt_star = G_star.t() ;
  const double       mu_star = arma::as_scalar( x.t() * delta ) ;
  const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
  
  // Init. the MC sample 
  arma::cube SMCret( TT, p, nSim );
  
  // SMC stuffs 
  int star_count = 0; 
  int star_times_minus_1 = Tstar.length() - 1;
  
  arma::mat P_F(p,p) ;
  
  arma::colvec FF(p);
  arma::rowvec FFt(p);
  arma::mat    FFtFF(p, p);
  
  arma::mat    P_P(p, p);
  arma::colvec yast_P(nSim);
  arma::colvec log_ws_IS_it(nSim);
  arma::colvec yast_(nSim);
  double S_P;
  
  arma::colvec xhat_col_vec(nSim);
  
  
  arma::uvec order_rss(nSim);
  arma::umat order_matrix(TT, nSim, arma::fill::value(TT));
  
  arma::cube subCube ;
  
  arma::vec Gamma ;
  
  //do{ 
  arma::mat xPart   = cpp_mvrnormArma_transposed( nSim, mu0, Sigma0) ;
  arma::mat xhat_F = arma::repmat(mu0, 1, nSim);
  arma::mat xhat_P(p, nSim);
  
  for(int tt = 0; tt < TT; ++tt) {
    
    FFt   = matFF.submat(tt, 0, tt, p-1) ;
    FF    = FFt.t();
    FFtFF = FF * FFt;
    
    if( y(tt) == NA_INTEGER){
      // Prior Evaluate evolutions
      if(tt != (Tstar(star_count)-1)){
        xhat_P  = G * xPart ;
        xPart   = xhat_P + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, SigmaEps) ;
      }else{ 
        xhat_P  = G_star * xPart ;
        xPart   = xhat_P + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, SigmaEps_star) ;
        if(star_times_minus_1 != star_count) star_count = star_count + 1;
      }  
    }else{ 
      // Evaluate Evolutions
      if(tt != (Tstar(star_count)-1)){
        xhat_P  = G * xPart;                      
        P_P     = SigmaEps ;  
      }else{   
        xhat_P  = G_star * xPart;                      
        P_P     = SigmaEps_star ;
        if(star_times_minus_1 != star_count) star_count = star_count + 1;
      }   
      
      yast_P  = (FFt * xhat_P).t();
      S_P     = arma::as_scalar(FFt*P_P*FF + 1.0);
      
      // Compute weights
      log_ws_IS_it = arma::log(
        arma::normcdf(  (2.0 * y(tt) - 1.0) * (yast_P + mu_star )
                          / std::sqrt(S_P) ) ) ; 
      
      log_ws_IS_it = log_ws_IS_it - arma::max(log_ws_IS_it);
      log_ws_IS_it.replace(arma::datum::nan, -100.0) ;
      
      // Resample step (y_{it} = 1)
      order_rss = rcpp_sample(x_order, nSim, arma::exp(log_ws_IS_it) );
      order_matrix.row(tt) = order_rss.t()  ;
      
      yast_P    = yast_P(order_rss);
      xhat_P    = xhat_P.cols(order_rss);
      
      switch( y(tt) ) {
      case 1 :
        for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
          yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + mu_star, 
                std::sqrt(S_P), 0, R_PosInf);
        } 
        break;
      case 0 : 
        for(int sample_to_evol = 0; sample_to_evol < nSim; ++sample_to_evol) {
          yast_(sample_to_evol) = r_truncnorm( yast_P(sample_to_evol) + mu_star, 
                std::sqrt(S_P), R_NegInf, 0);
        } 
        break;
      default: 
        Rcpp::Rcout << "Warnings: the ts is not binary" << std::endl;
      }
      
      // MC Evolution
      xhat_col_vec = yast_ - yast_P - mu_star   ;
      xhat_F       = xhat_P + P_P * FF * xhat_col_vec.t() / S_P ;
      
      P_F          = P_P - P_P * FFtFF * P_P / S_P;
      xPart        = xhat_F + cpp_mvrnormArma_transposed(nSim, zero_ex_ev, P_F) ;
      
    } 
    xPart.clamp( - 15.0, 15.0) ; // for stability
    SMCret.row(tt) = xPart ;
  }
  
  int old_pos = 0 ;
  int gen_pos;
  arma::mat RetMat(TT,p) ;
  for(int tt = 0; tt < TT-1 ; ++tt){
    RetMat.row( TT-tt-1) = SMCret.slice( old_pos ).row( TT-tt-1 ) ;
    gen_pos = order_matrix(TT-tt-1, old_pos) ;
    if( gen_pos != TT ){
      old_pos = gen_pos ;
    }
  }
  RetMat.row(0) = SMCret.slice(old_pos).row(0)   ;
  Gamma = arma::sum( RetMat % matFF, 1) ;
  return  Gamma ;
} 


//' Compute the summary statistics \eqn{R_i} from the state trajectory
//'
//' @description
//' Given the full latent state trajectory \eqn{\boldsymbol{\Theta}_i \in
//' \mathbb{R}^{T_i \times p}} and the observation design matrix \eqn{\mathbf{Z}_i
//' \in \mathbb{R}^{T \times p}}, this function returns the integer \eqn{R_i \in \{0, \ldots, 7\}}.
//'
//' For a version that accepts \eqn{\boldsymbol{\eta}} directly (avoiding
//' recomputation), see \code{\link{rcpp_get_Ri_wSplit_from_eta}}.
//'
//' @param theta \eqn{T_i \times p} matrix of latent states
//'   \eqn{\mathbf{x}_1, \ldots, \mathbf{x}_T} (one row per time point).
//' @param Zi \eqn{T \times p} observation design matrix for series \eqn{i}.
//'   Row \eqn{t} is \eqn{\mathbf{F}_t^\top}.
//' @param Tstar Numeric vector of 0-based split-point indices defining the
//'   last segment boundary used by criterion \eqn{R_3}.
//'
//' @return An integer in \eqn{\{0, \ldots, 7\}}.
//'
//' @seealso \code{\link{rcpp_get_Ri_wSplit_from_eta}},
//'   \code{\link{sample_Ri_under_g_wSplit}}
// [[Rcpp::export]]
int rcpp_get_Ri_wSplit(const arma::mat & theta,
                       const arma::mat & Zi,
                       const arma::vec & Tstar){
  
  // R1
  arma::vec eta = arma::sum(theta % Zi,1) ;
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    }
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    }
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count ) ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
    
  }
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int Tstart_pos ;
  
  if( Tstar.n_elem == 1){
    Tstart_pos = 0;
  }else{ 
    Tstart_pos = Tstar(Tstar.n_elem-2) -1  ;
  } 
  
  // R3
  arma::vec subvec = eta.subvec( Tstart_pos, Ti-1 );
  double pR3p1 = arma::mean( subvec );
  arma::vec subvec2 = eta.subvec( std::max( Tstart_pos - 90, 0), 
                                  std::max( Tstart_pos -  1, 0)  );
  double pR3p2 =  arma::mean( subvec2 ) ; // Tstar_pos = pos-1
  
  
  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  strRi += pR3p1 > pR3p2 ? '1' : '0';
  
  if ( strRi == "000") {
    return 0;
  } else if ( strRi == "001") { 
    return 1;
  } else if ( strRi == "010") { 
    return 2;
  } else if ( strRi == "011") { 
    return 3;
  } else if ( strRi == "100") { 
    return 4;
  } else if ( strRi == "101") { 
    return 5;
  } else if ( strRi == "110") { 
    return 6;
  } else { 
    return 7; 
  } 
  
}  

//' Compute \eqn{R_i} directly from the centered linear predictor.
//'
//' @description
//' Identical to \code{\link{rcpp_get_Ri_wSplit}} but accepts the centered
//' linear predictor vector.
//'
//' @param eta  Numeric vector of length \eqn{T}: the centered linear
//'   predictor \eqn{\eta_t = \mathbf{F}_t^\top \mathbf{x}_t}.
//' @param Tstar Numeric vector of 0-based split-point indices; see
//'   \code{\link{rcpp_get_Ri_wSplit}} for details.
//'
//' @return An integer in \eqn{\{0, \ldots, 7\}}.
//'
//' @seealso \code{\link{rcpp_get_Ri_wSplit}},
//'   \code{\link{rcpp_rCenteredLinearPredictor_RaoBlackwellAuxSMC}}
// [[Rcpp::export]]
int rcpp_get_Ri_wSplit_from_eta( const arma::vec eta, 
                                 const arma::vec Tstar){
  
  
  // R1
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    }
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    } 
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count ) ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
  } 
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int Tstart_pos ;
  
  if( Tstar.n_elem == 1){
    Tstart_pos = 0;
  }else{ 
    Tstart_pos = Tstar(Tstar.n_elem-2) -1  ;
  } 
  
  // R3
  arma::vec subvec = eta.subvec( Tstart_pos, Ti-1 );
  double pR3p1 = arma::mean( subvec ) ;
  arma::vec subvec2 = eta.subvec( std::max( Tstart_pos - 90, 0), 
                                  std::max( Tstart_pos -  1, 0)  );
  double pR3p2 =  arma::mean( subvec2 ) ; // Tstar_pos = pos-1
  
  
  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  strRi += pR3p1 > pR3p2 ? '1' : '0';
  
  if ( strRi == "000") {
    return 0;
  } else if ( strRi == "001") { 
    return 1;
  } else if ( strRi == "010") { 
    return 2;
  } else if ( strRi == "011") { 
    return 3;
  } else if ( strRi == "100") { 
    return 4;
  } else if ( strRi == "101") { 
    return 5;
  } else if ( strRi == "110") { 
    return 6;
  } else { 
    return 7; 
  }
}
 
//' Sample the prior predictive distribution of \eqn{R_i} under the state-space
//' prior \eqn{g}
//'
//' @description
//' Used to approximates the prior predictive distribution of the \eqn{R_i} by Monte Carlo simulation
//'  under the state-space prior \eqn{g} defined in the manuscript, Section 3.  For each of \code{nSim}
//' draws:
//' \enumerate{
//'   \item a full state trajectory
//'         \eqn{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T} is
//'         simulated from the prior;
//'   \item the risk index \eqn{R_i \in \{0, \ldots, 7\}} is computed via
//'         \code{\link{rcpp_get_Ri_wSplit}}.
//' }
//' The returned vector of \code{nSim} draws can be used to estimate
//' \eqn{p(R_i = r \mid g)} for each \eqn{r}, enabling prior elicitation
//' and sensitivity analysis as described in the paper.
//'
//' @param TT    Length of the time series \eqn{T_i}.
//' @param matFF \eqn{T_i \times p} observation design matrix (same role as
//'   \code{Zi} in \code{\link{rcpp_get_Ri_wSplit}}).
//' @param mu0   Prior mean of the initial state.
//' @param Sigma0 Prior covariance of \eqn{\mathbf{x}_0} (\eqn{p \times p}).
//' @param G        Baseline transition matrix (\eqn{p \times p}).
//' @param G_star   Alternative-regime transition matrix (\eqn{p \times p}).
//' @param SigmaEps      Baseline evolution covariance (\eqn{p \times p}).
//' @param SigmaEps_star Alternative evolution covariance
//'   (\eqn{p \times p}).
//' @param Tstar  Numeric vector of \strong{0-based} split-point indices
//'   at which the alternative regime is applied.
//' @param nSim  Number of Monte Carlo draws.  Default: \code{10000}.
//'
//' @return A numeric vector of length \code{nSim} with values in
//'   \eqn{\{0, \ldots, 7\}}, representing independent draws from the prior
//'   predictive distribution of \eqn{R_i} under \eqn{g}.
//'
//' @seealso \code{\link{rcpp_get_Ri_wSplit}},
//'   \code{\link{rcpp_get_Ri_wSplit_from_eta}}
//'
//' @examples
//' \dontrun{
//' draws <- sample_Ri_under_g_wSplit(
//'   TT          = 365L,
//'   matFF       = matFF,
//'   mu0         = rep(0, p),
//'   Sigma0      = diag(p),
//'   G           = G,
//'   G_star      = G_star,
//'   SigmaEps    = SigE,
//'   SigmaEps_star = SigE_s,
//'   Tstar       = c(90, 365),
//'   nSim        = 10000L
//' )
//' }
// [[Rcpp::export]]
arma::vec sample_Ri_under_g_wSplit(
     const int TT,
     const arma::mat & matFF,
     const arma::colvec & mu0,
     const arma::mat & Sigma0,
     const arma::mat & G,
     const arma::mat & G_star, 
     const arma::mat & SigmaEps,
     const arma::mat & SigmaEps_star,
     const arma::vec & Tstar,
     const int nSim = 10000){
   
   const int p = mu0.n_elem ;
   arma::vec theta_it(p);
   
   const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
   
   int star_count = 0;
   int star_times_minus_1 = Tstar.n_elem - 1;
   
   arma::mat Theta(TT, p) ;
   arma::vec Ris(nSim) ;
   
   for (int sim = 0; sim < nSim; ++sim){
     theta_it = mvrnormArma1( mu0, Sigma0) ;
     
     for (int tt = 0; tt < TT; ++tt){
       
       if(tt != Tstar(star_count)){
         theta_it = G * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps) ;
       }else{ 
         theta_it = G_star * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps_star) ;      
         if(star_times_minus_1 != star_count) star_count = star_count + 1; 
       }
       Theta.row(tt) = theta_it.t() ;
     } 
     Ris(sim) = rcpp_get_Ri_wSplit(Theta, matFF, Tstar) ;
   }
   return Ris ;
 } 

//' Sample from the prior predictive distribution a realization for the state-space parameters.
//'
//' @description
//' Used to approximates the prior predictive distribution of the \eqn{R_i} by Monte Carlo simulation
//'  under the state-space prior \eqn{g} defined in the manuscript, Section 3.  For each of \code{nSim}
//' draws:
//' \enumerate{
//'   \item a full state trajectory
//'         \eqn{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T} is
//'         simulated from the prior;
//'   \item the risk index \eqn{R_i \in \{0, \ldots, 7\}} is computed via
//'         \code{\link{rcpp_get_Ri_wSplit}}.
//' }
//' The returned vector of \code{nSim} draws can be used to estimate
//' \eqn{p(R_i = r \mid g)} for each \eqn{r}, enabling prior elicitation
//' and sensitivity analysis as described in the paper.
//'
//' @param TT    Length of the time series \eqn{T_i}.
//' @param matFF \eqn{T_i \times p} observation design matrix (same role as
//'   \code{Zi} in \code{\link{rcpp_get_Ri_wSplit}}).
//' @param mu0   Prior mean of the initial state.
//' @param Sigma0 Prior covariance of \eqn{\mathbf{x}_0} (\eqn{p \times p}).
//' @param G        Baseline transition matrix (\eqn{p \times p}).
//' @param G_star   Alternative-regime transition matrix (\eqn{p \times p}).
//' @param SigmaEps      Baseline evolution covariance (\eqn{p \times p}).
//' @param SigmaEps_star Alternative evolution covariance
//'   (\eqn{p \times p}).
//' @param Tstar  Numeric vector of \strong{0-based} split-point indices
//'   at which the alternative regime is applied.
//' @param nSim  Number of Monte Carlo draws.  Default: \code{10000}.
//'
//' @return A numeric vector of length \code{nSim} with values in
//'   \eqn{\{0, \ldots, 7\}}, representing independent draws from the prior
//'   predictive distribution of \eqn{R_i} under \eqn{g}.
//'
//' @seealso \code{\link{rcpp_get_Ri_wSplit}},
//'   \code{\link{rcpp_get_Ri_wSplit_from_eta}}
//'
//' @examples
//' \dontrun{
//' draws <- sample_Ri_under_g_wSplit(
//'   TT          = 365L,
//'   matFF       = matFF,
//'   mu0         = rep(0, p),
//'   Sigma0      = diag(p),
//'   G           = G,
//'   G_star      = G_star,
//'   SigmaEps    = SigE,
//'   SigmaEps_star = SigE_s,
//'   Tstar       = c(90, 365),
//'   nSim        = 10000L
//' )
//' }
// [[Rcpp::export]]
arma::ivec sample_biniary_ts_wSplit_from_G( const double         mu,
                                            const int            TT,
                                            const arma::mat    & matFF,
                                            const arma::colvec & mu0,
                                            const arma::mat    & Sigma0,
                                            const arma::mat    & G,
                                            const arma::mat    & G_star, 
                                            const arma::mat    & SigmaEps,
                                            const arma::mat    & SigmaEps_star,
                                            const arma::vec    & Tstar ){
  
  const int p = mu0.n_elem ;
  arma::vec theta_it(p);
  
  const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
  
  int star_count = 0;
  int star_times_minus_1 = Tstar.n_elem - 1;
  
  arma::mat Theta(TT, p) ;
  arma::ivec BinTS(TT) ;
  
  theta_it = mvrnormArma1( mu0, Sigma0) ;
  
  for (int tt = 0; tt < TT; ++tt){
    
    if(tt != Tstar(star_count)){
      theta_it = G * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps) ;
    }else{ 
      theta_it = G_star * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps_star) ;      
      if(star_times_minus_1 != star_count) star_count = star_count + 1; 
    }
    Theta.row(tt) = theta_it.t() ;
  }
  arma::vec probs = arma::normcdf( mu + arma::sum(Theta % matFF,1));
  arma::vec u     = arma::randu(probs.n_elem) ;
  
  BinTS =  arma::conv_to<arma::ivec>::from(probs > u);
  return  BinTS;
} 

// ============================================================
//  Updates steps for MCMC
// ============================================================

arma::mat rcpp_update_xi_star( const Rcpp::List & state,
                               const arma::colvec & a ){
  
  arma::mat    Old_xi = Rcpp::as<arma::mat>(  state["xi"] ) ;
  arma::colvec Ri_vec = Rcpp::as<arma::vec>(  state["Ri"])  ;
  arma::colvec rho_vec = Rcpp::as<arma::vec>( state["rho"])  ;
  
  int H = Old_xi.n_rows ; 
  int Q = Old_xi.n_cols ;
  int n = Ri_vec.n_elem ;
  arma::vec new_parm ;
  arma::mat mat_counts_vect(H,Q) ;
  int cRi;
  int crhoi;
  for( int i = 0; i < n; i++){
    crhoi = rho_vec(i) ;
    cRi = Ri_vec(i) ;
    mat_counts_vect( crhoi, cRi) += 1 ;
  }
  
  for( int h = 0; h < H; h++){
    new_parm      = a + mat_counts_vect.row(h).as_col() ;
    Old_xi.row(h) = cpp_rdirichletArma1( new_parm ).as_row() ;
  } 
  return Old_xi ;
} 

Rcpp::List rcpp_update_Ri_gamma_and_rho(
    const Rcpp::List & data,
    const Rcpp::List & state, 
    arma::ivec & gTable,
    const Rcpp::CharacterVector & IDs_int,
    const arma::vec & a,
    const double M,
    const double sigma,
    const arma::vec &  mu0_int,
    const arma::mat & Sigma0_int,
    const arma::mat & G_int,
    const arma::mat & G_star_int, 
    const arma::mat & SigmaEps_int,
    const arma::mat & SigmaEps_star_int,
    const arma::mat & X_int,
    const arma::mat & gl_matrix,
    int AR){
  
  // inizialization
  int n_subject = IDs_int.size() ;
  
  Rcpp::List Theta_list_old = state["Gamma"] ;
  Rcpp::List Theta_list_ret(n_subject); 
  Rcpp::List SubjectData ;
  
  const arma::colvec delta_int = state["delta"] ;
  arma::ivec rho = state["rho"] ;
  arma::mat  xi  = state["xi"]  ;
  arma::colvec Ri_vec = state["Ri"] ;
  
  int H = xi.n_rows ;
  
  int old_rho;
  int new_rho;
  
  arma::vec    Ex_new =  a / arma::sum( a );
  arma::colvec new_xi_star ;
  arma::colvec old_xi_star ;
  arma::colvec rho_probs ;
  
  arma::colvec probs_h ;
  arma::colvec probs_h_old_Ri ;
  arma::uvec   labels ;
  
  // Place Holders for subjects
  Rcpp::IntegerVector ts_it ;
  arma::colvec        Tstar_it ;
  arma::mat           Zi_it ;
  arma::colvec        x_it ;
  
  int old_Ri_it ;
  int new_Ri_it ;
  
  double alpha_it ; 
  double U_it; 
  
  arma::vec gli_it;
  arma::mat newTheta_it ;
  
  // Subject Loop 
  for (int subject = 0; subject < n_subject; ++subject){
    
    // load r.v. for the iteration
    SubjectData = data[ subject ] ;
    ts_it       = Rcpp::as<Rcpp::IntegerVector>(SubjectData["ts"]) ;
    Zi_it       = Rcpp::as<arma::mat>(SubjectData["Z"]) ;
    Tstar_it    = Rcpp::as<arma::colvec>(SubjectData["Tstar"]) ;
    x_it        = X_int.row( subject ).t() ;
    gli_it      = gl_matrix.row(subject).t() ;
    
    // old Ri= rl and old rho = h
    old_Ri_it = Ri_vec(subject);
    old_rho   = rho(subject);
    
    //new Ri= rl' from g(\\theta_i)
    newTheta_it = rcpp_rEta_conditioned_RaoBlackwellAuxSMC(
      ts_it, delta_int , x_it, Zi_it,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int , 
      SubjectData["Tstar"]);
    new_Ri_it = rcpp_get_Ri_wSplit_from_eta(newTheta_it, Tstar_it);
    
    // remove the rho \xi* and \rho from the State
    if( gTable(old_rho) == 1){
      old_xi_star  = xi.row(old_rho).t();
      gTable       = rcpp_remove_element( gTable, old_rho) ;
      xi           = rcpp_remove_row( xi, old_rho) ;
      rho          = rcpp_correct_labels( rho, old_rho) ;
       
      H            -= 1 ;
      old_rho      = H  ; 
    }else{  
      gTable(old_rho) -= 1 ;
    }     
    
    labels = rcpp_index_gen( H+1 ) ;
    
    probs_h.resize(H + 1);
    probs_h.subvec(0,H-1)         = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(new_Ri_it) ; 
    probs_h(H)                    = Ex_new(new_Ri_it) * ( M + H * sigma);
    
    new_rho = rcpp_sample(labels,1, probs_h)(0);
    
    probs_h_old_Ri.resize(H+1) ; 
    probs_h_old_Ri.subvec(0,H-1)  = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(old_Ri_it) ; 
    probs_h_old_Ri(H)             = Ex_new(old_Ri_it) * ( M + H * sigma);
    
    alpha_it = arma::sum( probs_h ) / arma::sum(probs_h_old_Ri) * gli_it(old_Ri_it) / gli_it(new_Ri_it) ;
    U_it     = arma::randu() ;
    
    if( alpha_it > U_it){
      
      // Accpeted transition
      Theta_list_ret[subject] = newTheta_it ; 
      Ri_vec(subject)         = new_Ri_it  ;
      rho(subject)            = new_rho ;
      
      AR += 1 ; 
      
      if(new_rho == H){
        rcpp_add_elem_1(gTable);
        new_xi_star = cpp_rdirichletArma1_plus_1_pos(a, new_Ri_it) ;
        rcpp_add_row(xi, new_xi_star) ;
        H += 1 ;
      }else{
        gTable(new_rho) += 1 ;
        }
    }else{ 
      
      Theta_list_ret[subject] = Theta_list_old[subject] ;
      rho(subject) = old_rho ; // needed for singletons
       
      if(old_rho == H){
        rcpp_add_elem_1(gTable);
        rcpp_add_row(xi, old_xi_star) ;
        H += 1 ;
      }else{ gTable(old_rho) += 1 ; } 
    } 
  } 
  
  Theta_list_ret.attr("names") = IDs_int ;
  
  return Rcpp::List::create(Rcpp::Named("up_AR")     = AR,
                            Rcpp::Named("up_Ri")     = Ri_vec,
                            Rcpp::Named("up_Gamma")  = Theta_list_ret,
                            Rcpp::Named("up_table")  = gTable, 
                            Rcpp::Named("up_xi")     = xi,
                            Rcpp::Named("up_rho")    = rho);
}  


arma::colvec rcpp_update_delta_v2( const Rcpp::List & state,
                                   const int n,
                                   const arma::colvec & y, 
                                   const arma::mat & X,
                                   const arma::colvec & lower,
                                   const arma::colvec & upper,
                                   const Rcpp::List & Indeces,
                                   const arma::mat & Prec0,
                                   const arma::vec & d0,
                                   const arma::mat & V){
  
  Rcpp::List ALLg = state["Gamma"] ;
  arma::vec  old_delta = Rcpp::as<arma::vec>( state["delta"] ) ;
  arma::colvec Gamma    ;
  arma::colvec gamma_it ;
  arma::colvec m ;
  arma::colvec z ; 
  arma::colvec u(X.n_rows) ;
  
  arma::colvec old_lin_pred ;
  
  Gamma.set_size(0)      ;
  for( int subject = 0; subject < n; ++subject ){
    gamma_it = Rcpp::as<arma::vec>( ALLg[ subject ]).elem( Rcpp::as<arma::uvec>( Indeces[subject] ) ) ;
    Gamma    = arma::join_cols( Gamma, gamma_it ) ;
  }
  
  old_lin_pred = X * old_delta + Gamma ;
  
  z = rcpp_rtnorm_vec( old_lin_pred, lower, upper ) ;
  
  m = V * ( Prec0 * d0 + X.t() * ( z - Gamma )) ; 
  
  if( m.has_nan() ){
    Rcpp::Rcout << "numerical instability" << std::flush ;
    return old_delta ;
  }else{
    return cpp_mvrnormArma1(m, V) ;
  }
}

//' Collapsed Gibbs sampler for the BNP mxiture of SUN models (full output)
//'
//' @description
//' Runs a collapsed Gibbs sampler for the model described in the paper. 
//'
//' For a memory-efficient variant that stores only
//' \eqn{(\boldsymbol{\delta}, \mathbf{R})} see
//' \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}}.
//'
//' @param data
//'   Named \code{Rcpp::List} of length \eqn{n}.  Each element is itself a
//'   list with at least a field \code{"ts"} (the binary time series for
//'   subject \eqn{i}) and any covariates needed by the update functions.
//' @param X
//'   \eqn{n \times q} matrix of subject-level static covariates
//'   \eqn{\mathbf{x}_i}.  Used to build the stacked probit design matrix
//'   and forwarded to the state-trajectory update.
//' @param IDs
//'   Character vector of length \eqn{n} with subject identifiers.
//' @param STATE
//'   Named \code{Rcpp::List} holding the current MCMC state.  Must contain
//'   at least the fields \code{"rho"}, \code{"delta"}, \code{"xi"},
//'   \code{"Gamma"}, and \code{"Ri"}.  Modified in place at each sweep.
//' @param Prior
//'   Named list of hyperparameters for the DP and probit blocks:
//'   \describe{
//'     \item{\code{"a"}}{Concentration vector \eqn{\mathbf{a}} of the
//'       base-measure Dirichlet prior on \eqn{\boldsymbol{\xi}^*}.}
//'     \item{\code{"M"}}{DP precision parameter \eqn{M > 0}.}
//'     \item{\code{"sigma"}}{Discount parameter \eqn{\sigma}.}
//'     \item{\code{"d0"}}{Prior mean vector for \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{"D0"}}{Diagonal of the prior precision matrix
//'       \eqn{\mathbf{D}_0} for \eqn{\boldsymbol{\delta}}.}
//'   }
//' @param gPrior
//'   Named list of state-space prior hyperparameters for the DLM prior
//'   \eqn{g}:
//'   \describe{
//'     \item{\code{"m0"}, \code{"S0"}}{Prior mean and covariance of the
//'       initial state \eqn{\mathbf{x}_0}.}
//'     \item{\code{"G1"}, \code{"G2"}}{Baseline and alternative-regime
//'       transition matrices \eqn{\mathbf{G}} and \eqn{\mathbf{G}^*}.}
//'     \item{\code{"Ve"}, \code{"Ves"}}{Baseline and alternative-regime
//'       innovation covariances
//'       \eqn{\boldsymbol{\Sigma}_\varepsilon} and
//'       \eqn{\boldsymbol{\Sigma}_\varepsilon^*}.}
//'     \item{\code{"gli"}}{\eqn{8 \times K} matrix mapping each risk
//'       category \eqn{R_i \in \{0,\ldots,7\}} to a cluster probability
//'       vector.}
//'   }
//' @param sample    Number of posterior draws to store.
//'   Default: \code{2000}.
//' @param thinning  Store one sample every \code{thinning} post-burn-in.
//'   Default: \code{10}.
//' @param burn      Number of burn-in.
//'   Default: \code{1000}.
//' @param nSim      Number of SMC particles forwarded to the trajectory
//'   sampler inside \code{rcpp_update_Ri_gamma_and_rho}.
//'   Default: \code{100}.
//'
//' @return A named \code{Rcpp::List} with five fields, each of length
//'   \code{sample}:
//'   \describe{
//'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
//'       of the probit coefficient vector \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{rho}}{\eqn{\texttt{sample} \times n} matrix of cluster
//'       assignment draws \eqn{\boldsymbol{\rho}}.}
//'     \item{\code{xi_star}}{List of \code{sample} draws of the cluster
//'       atom matrix \eqn{\boldsymbol{\xi}^*}.}
//'     \item{\code{Gamma}}{List of \code{sample} draws of the latent state
//'       trajectories \eqn{\boldsymbol{\Gamma}}.}
//'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
//'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
//'   }
//'
//' @seealso
//'   \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}},
//'   \code{\link{rcpp_get_Ri_wSplit}},
//'   \code{\link{rcpp_rCenteredLinearPredictor_RaoBlackwellAuxSMC}}
//'
//' @examples
//' \dontrun{
//' fit <- SUN_gibbs_collapsed_sampler_v2(
//'   data     = subject_list,
//'   X        = X_mat,
//'   IDs      = ids,
//'   STATE    = initial_state,
//'   Prior    = prior_list,
//'   gPrior   = g_prior_list,
//'   sample   = 2000L,
//'   thinning = 10L,
//'   burn     = 1000L,
//'   nSim     = 100L
//' )
// [[Rcpp::export]]
Rcpp::List SUN_gibbs_collapsed_sampler_v2( const Rcpp::List data,
                                           const arma::mat X,
                                           const Rcpp::CharacterVector IDs,
                                           Rcpp::List STATE,
                                           const Rcpp::List  Prior,
                                           const Rcpp::List gPrior,
                                           const int sample = 2000,
                                           const int thinning = 10,
                                           const int burn = 1000,
                                           const int nSim = 100){
  
  std::string msg = "Creating Constant for the MCMC...\n" ;
  Rcpp::Rcout << msg << std::flush ;
  
  const int n = IDs.size();
  
  const  arma::vec a = Prior["a"];
  double M = Prior["M"];
  double sigma = Prior["sigma"];
  arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
  arma::ivec gTable_int   = rcpp_arma_table(sg);
  
  const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
  const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
  const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
  const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]); 
  const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
  const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
  const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
  arma::colvec delta_it ; 
  
  const int q = X.n_cols ;       // x_i
  const int p = mu0_int.n_elem ; // z_itK
  
  arma::mat  return_rho(sample,  n);
  arma::mat  return_delta(sample, q) ;
  Rcpp::List return_xi_star(sample) ;
  Rcpp::List return_theta_list(sample) ;
  Rcpp::List return_Ri_list(sample) ;
  
  Rcpp::List list_of_updates ;
  
  Rcpp::IntegerVector y_it ; 
  arma::vec xi_it ;
  arma::mat Zi_it ;
  Rcpp::IntegerVector Tstar_it ;
  double Ti_it ;  
  
  int AR_thetas ;
  int AR_delta ;
  
  // For Probit data augmentation 
  Rcpp::List Indeces_for_obs( n ) ;
  
  // DV 
  Rcpp::List SubjectData_intern;
  arma::vec y_intern ; 
  arma::rowvec xi_intern ;
  arma::uvec pos ;
   
  arma::mat    X_probit ;
  arma::colvec y_probit ;
  y_probit.set_size( 0 ) ; 
  
  const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
  const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
  
  for ( int subject = 0; subject < n; ++subject ) {
    
    SubjectData_intern  = data[ subject ] ;
     
    y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
    xi_intern = X.row(subject)  ;
    
    pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
    
    Indeces_for_obs[subject] = pos ;
    
    switch( subject ) {
    case 0:
      X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
      break;
    default: 
      X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
    }
    
    y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
  }  
  
  arma::colvec lower_limits( y_probit.n_elem ) ;
  arma::colvec upper_limits( y_probit.n_elem ) ;
  
  lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
  lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
  
  upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
  upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
  
  arma::mat invD0 = arma::inv( D0 ) ;
  arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
  
  
  // Gibbs
  
  int SampleStored = 0;
  const int MaxIteration = burn + sample * thinning ;
   
  // MCMC
  msg = "Running the chain...\n";
  Rcpp::Rcout << msg << std::endl;
  
  for (int it = 0; it < MaxIteration; ++it) {
    
    Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
    
    // MCMC CORE 
    // Theta {i=1, ... , N}
    list_of_updates = rcpp_update_Ri_gamma_and_rho(
      data, STATE, gTable_int,
      IDs, a, M, sigma,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int,
      X, gl_matrix,
      AR_thetas) ; 
    
    STATE["Gamma"] = list_of_updates["up_Gamma"] ;
    STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
    STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
    STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
    gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
    AR_thetas      = list_of_updates["up_AR"]    ;
    
    // Xi moving
    STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
    
    // delta
    STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                      y_probit, X_probit, 
                                      lower_limits, upper_limits,
                                      Indeces_for_obs, 
                                      invD0, d0, V ) ;
    
    // STORE 
    if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
      return_rho.row(SampleStored)    = Rcpp::as<arma::rowvec>(STATE["rho"])   ;
      return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
      return_xi_star[SampleStored]    = STATE["xi"]    ;
      return_Ri_list[SampleStored]    = STATE["Ri"]    ;
      return_theta_list[SampleStored] = STATE["Gamma"] ;
      
      SampleStored = SampleStored + 1;
    }
    
    Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
    Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
    Rcpp::Rcout <<  gTable_int << std::flush;
  } 
  
  Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
  
  return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                             Rcpp::Named("rho")     = return_rho,
                             Rcpp::Named("xi_star") = return_xi_star,
                             Rcpp::Named("Gamma")   = return_theta_list,
                             Rcpp::Named("Ri")      = return_Ri_list );
} 



//' Collapsed Gibbs sampler for the SUN model (lightweight output)
//'
//' @description
//' Memory-efficient variant of
//' \code{\link{SUN_gibbs_collapsed_sampler_v2}}.  
//'
//' Use this variant when:
//' \itemize{
//'   \item only posterior inference on \eqn{\boldsymbol{\delta}} and the
//'     summaries statistics classification \eqn{p(R_i = r \mid \mathbf{y})} is needed;
//'   \item the number of subjects \eqn{n} or series length \eqn{T} makes
//'     storing full trajectories prohibitive.
//' }
//'
//' All parameters are identical to
//' \code{\link{SUN_gibbs_collapsed_sampler_v2}}; see that page for full
//' descriptions.
//'
//' @inheritParams SUN_gibbs_collapsed_sampler_v2
//'
//' @return A named \code{Rcpp::List} with two fields:
//'   \describe{
//'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
//'       of \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
//'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
//'   }
//'
//' @seealso \code{\link{SUN_gibbs_collapsed_sampler_v2}}
//'
//' @examples
//' \dontrun{
//' }
// [[Rcpp::export]]
Rcpp::List SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta( const Rcpp::List data,
                                                              const arma::mat X,
                                                              const Rcpp::CharacterVector IDs,
                                                              Rcpp::List STATE,
                                                              const Rcpp::List  Prior,
                                                              const Rcpp::List gPrior,
                                                              const int sample = 2000,
                                                              const int thinning = 10,
                                                              const int burn = 1000,
                                                              const int nSim = 100){
  
  std::string msg = "Creating Constant for the MCMC...\n" ;
  Rcpp::Rcout << msg << std::flush ;
  
  const int n = IDs.size();
  
  const  arma::vec a = Prior["a"];
  double M = Prior["M"];
  double sigma = Prior["sigma"];
  arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
  arma::ivec gTable_int   = rcpp_arma_table(sg);
  
  const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
  const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
  const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
  const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]);  
  const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
  const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
  const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
  arma::colvec delta_it ; 
  
  const int q = X.n_cols ;       // x_i
  const int p = mu0_int.n_elem ; // z_itK
  
  arma::mat  return_delta(sample, q) ;
  Rcpp::List return_Ri_list(sample) ;
  
  Rcpp::List list_of_updates ;
  
  Rcpp::IntegerVector y_it ; 
  arma::vec xi_it ;
  arma::mat Zi_it ;
  Rcpp::IntegerVector Tstar_it ;
  double Ti_it ;  
  
  int AR_thetas ;
  int AR_delta ;
  
  // For Probit data augmentation 
  Rcpp::List Indeces_for_obs( n ) ;
  
  // DV 
  Rcpp::List SubjectData_intern;
  arma::vec y_intern ; 
  arma::rowvec xi_intern ;
  arma::uvec pos ;
  
  arma::mat    X_probit ;
  arma::colvec y_probit ;
  y_probit.set_size( 0 ) ; 
  
  const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
  const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
  
  for ( int subject = 0; subject < n; ++subject ) {
    
    SubjectData_intern  = data[ subject ] ;
    
    y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
    xi_intern = X.row(subject)  ;
    
    pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
    
    Indeces_for_obs[subject] = pos ;
    
    switch( subject ) {
    case 0:
      X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
      break;
    default:
      X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
    }
    
    y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
  }
  
  arma::colvec lower_limits( y_probit.n_elem ) ;
  arma::colvec upper_limits( y_probit.n_elem ) ;
  
  lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
  lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
  
  upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
  upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
  
  arma::mat invD0 = arma::inv( D0 ) ;
  arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
  
  
  // Gibbs
  
  int SampleStored = 0;
  const int MaxIteration = burn + sample * thinning ;
  
  // MCMC
  msg = "Running the chain...\n";
  Rcpp::Rcout << msg << std::endl;
  
  for (int it = 0; it < MaxIteration; ++it) {
    
    Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
    
    // MCMC CORE 
    
    // Theta {i=1, ... , N}
    list_of_updates = rcpp_update_Ri_gamma_and_rho(
      data, STATE, gTable_int,
      IDs, a, M, sigma,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int,
      X, gl_matrix,
      AR_thetas) ;
    
    STATE["Gamma"] = list_of_updates["up_Gamma"] ;
    STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
    STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
    STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
    gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
    AR_thetas      = list_of_updates["up_AR"]    ;
    
    // Xi moving
    STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
    
    // delta
    STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                      y_probit, X_probit, 
                                      lower_limits, upper_limits,
                                      Indeces_for_obs, 
                                      invD0, d0, V ) ;
    
    // STORE
    if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
      return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
      return_Ri_list[SampleStored]    = STATE["Ri"]    ;
      
      SampleStored = SampleStored + 1;
    }
    // Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
    // Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
    // Rcpp::Rcout <<  gTable_int << std::flush;
  } 
  
  Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
  
  return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                             Rcpp::Named("Ri")      = return_Ri_list );
}  



// ============================================================
//  Revision functions: S1 < 16864 < S2 
// ============================================================

// [[Rcpp::export]]
int rcpp_get_Ri_wSplit_v3(const arma::mat & theta,
                          const arma::mat & Zi,
                          const arma::vec & Tstar){
  
  arma::vec eta = arma::sum(theta % Zi,1) ;
  // R1
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count  = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    } 
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    }  
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count )-1 ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
  }  
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int T_starPost ;
  
  if( Tstar.n_elem == 1){
    T_starPost = 0;
  }else{ 
    T_starPost = Tstar(Tstar.n_elem-2) ;
  } 
  
  // R3
  int n = Ti - T_starPost;
  arma::vec x = arma::regspace(0, n-1);
  arma::vec y = eta.subvec(T_starPost, Ti-1);
  
  double x_mean = arma::mean(x);
  double y_mean = arma::mean(y);
  
  double numerator   = arma::sum((x - x_mean) % (y - y_mean));
  double denominator = arma::sum(arma::square(x - x_mean));
  
  double pR3 = numerator / denominator;
  
  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  strRi += pR3   >  0    ? '1' : '0';
  
  if ( strRi == "000") {
    return 0;
  } else if ( strRi == "001") { 
    return 1;
  } else if ( strRi == "010") { 
    return 2;
  } else if ( strRi == "011") { 
    return 3;
  } else if ( strRi == "100") { 
    return 4;
  } else if ( strRi == "101") { 
    return 5;
  } else if ( strRi == "110") { 
    return 6;
  } else { 
    return 7; 
  }
}

// [[Rcpp::export]]
int rcpp_get_Ri_wSplit_from_eta_v3(const arma::vec eta, 
                                   const arma::vec Tstar){
  
  // R1
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count  = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    } 
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    }  
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count )-1 ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
  }  
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int T_starPost ;
  
  if( Tstar.n_elem == 1){
    T_starPost = 0;
  }else{ 
    T_starPost = Tstar(Tstar.n_elem-2) ;
  } 
  
  // R3
  int n = Ti - T_starPost;
  arma::vec x = arma::regspace(0, n-1);
  arma::vec y = eta.subvec(T_starPost, Ti-1);
  
  double x_mean = arma::mean(x);
  double y_mean = arma::mean(y);
  
  double numerator   = arma::sum((x - x_mean) % (y - y_mean));
  double denominator = arma::sum(arma::square(x - x_mean));
  
  double pR3 = numerator / denominator;
  
  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  strRi += pR3   >  0    ? '1' : '0';

  if ( strRi == "000") {
     return 0;
  } else if ( strRi == "001") { 
     return 1;
  } else if ( strRi == "010") { 
    return 2;
  } else if ( strRi == "011") { 
    return 3;
  } else if ( strRi == "100") { 
    return 4;
  } else if ( strRi == "101") { 
    return 5;
  } else if ( strRi == "110") { 
    return 6;
  } else { 
    return 7; 
  }
}



// [[Rcpp::export]]
 arma::vec sample_Ri_under_g_wSplit_v3(
     const int TT,
     const arma::mat    & matFF,
     const arma::colvec & mu0,
     const arma::mat    & Sigma0,
     const arma::mat    & G,
     const arma::mat    & G_star, 
     const arma::mat    & SigmaEps,
     const arma::mat    & SigmaEps_star,
     const arma::vec    & Tstar,
     const int nSim  = 10000 ){
   
   const int p = mu0.n_elem ;
   arma::vec theta_it(p);
   
   const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
   
   int star_count = 0;
   int star_times_minus_1 = Tstar.n_elem - 1;
   
   arma::mat Theta(TT, p) ;
   arma::vec Ris(nSim) ;
   
   for (int sim = 0; sim < nSim; ++sim){
     theta_it = mvrnormArma1( mu0, Sigma0) ;
     
     for (int tt = 0; tt < TT; ++tt){
       
       if(tt != Tstar(star_count)){
         theta_it = G * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps) ;
       }else{ 
         theta_it = G_star * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps_star) ;      
         if(star_times_minus_1 != star_count) star_count = star_count + 1; 
       }
       Theta.row(tt) = theta_it.t() ;
     } 
     Ris(sim) = rcpp_get_Ri_wSplit_v3(Theta, matFF, Tstar) ;
   }
   return Ris ;
 } 

Rcpp::List rcpp_update_Ri_gamma_and_rho_v3(
    const Rcpp::List & data,
    const Rcpp::List & state, 
    arma::ivec & gTable,
    const Rcpp::CharacterVector & IDs_int,
    const arma::vec & a,
    const double M,
    const double sigma,
    const arma::vec &  mu0_int,
    const arma::mat & Sigma0_int,
    const arma::mat & G_int,
    const arma::mat & G_star_int, 
    const arma::mat & SigmaEps_int,
    const arma::mat & SigmaEps_star_int,
    const arma::mat & X_int,
    const arma::mat & gl_matrix,
    int AR){
  
  // inizialization
  int n_subject = IDs_int.size() ;
  
  Rcpp::List Theta_list_old = state["Gamma"] ;
  Rcpp::List Theta_list_ret(n_subject); 
  Rcpp::List SubjectData ;
  
  const arma::colvec delta_int = state["delta"] ;
  arma::ivec rho = state["rho"] ;
  arma::mat  xi  = state["xi"]  ;
  arma::colvec Ri_vec = state["Ri"] ;
  
  int H = xi.n_rows ;
  
  int old_rho;
  int new_rho;
  
  arma::vec    Ex_new =  a / arma::sum( a );
  arma::colvec new_xi_star ;
  arma::colvec old_xi_star ;
  arma::colvec rho_probs ;
  
  arma::colvec probs_h ;
  arma::colvec probs_h_old_Ri ;
  arma::uvec   labels ;
  
  // Place Holders for subjects
  Rcpp::IntegerVector ts_it ;
  arma::colvec        Tstar_it ;
  arma::mat           Zi_it ;
  arma::colvec        x_it ;
  
  int old_Ri_it ;
  int new_Ri_it ;
  
  double alpha_it ; 
  double U_it; 
  
  arma::vec gli_it;
  arma::mat newTheta_it ;
  
  // Subject Loop 
  for (int subject = 0; subject < n_subject; ++subject){
    
    // load r.v. for the iteration
    SubjectData = data[ subject ] ;
    ts_it       = Rcpp::as<Rcpp::IntegerVector>(SubjectData["ts"]) ;
    Zi_it       = Rcpp::as<arma::mat>(SubjectData["Z"]) ;
    Tstar_it    = Rcpp::as<arma::colvec>(SubjectData["Tstar"]) ;
    x_it        = X_int.row( subject ).t() ;
    gli_it      = gl_matrix.row(subject).t() ;
    
    // old Ri= rl and old rho = h
    old_Ri_it = Ri_vec(subject);
    old_rho   = rho(subject);
    
    //new Ri= rl' from g(\\theta_i)
    newTheta_it = rcpp_rEta_conditioned_RaoBlackwellAuxSMC(
      ts_it, delta_int , x_it, Zi_it,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int , 
      SubjectData["Tstar"]);
    new_Ri_it = rcpp_get_Ri_wSplit_from_eta_v3(newTheta_it, Tstar_it);
    
    // remove the rho \xi* and \rho from the State
    if( gTable(old_rho) == 1){
      old_xi_star  = xi.row(old_rho).t();
      gTable       = rcpp_remove_element( gTable, old_rho) ;
      xi           = rcpp_remove_row( xi, old_rho) ;
      rho          = rcpp_correct_labels( rho, old_rho) ;
       
      H            -= 1 ;
      old_rho      = H  ; 
    }else{  
      gTable(old_rho) -= 1 ;
    }     
    
    labels = rcpp_index_gen( H+1 ) ;
    
    probs_h.resize(H + 1);
    probs_h.subvec(0,H-1)         = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(new_Ri_it) ; 
    probs_h(H)                    = Ex_new(new_Ri_it) * ( M + H * sigma);
    
    new_rho = rcpp_sample(labels,1, probs_h)(0);
    
    probs_h_old_Ri.resize(H+1) ; 
    probs_h_old_Ri.subvec(0,H-1)  = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(old_Ri_it) ; 
    probs_h_old_Ri(H)             = Ex_new(old_Ri_it) * ( M + H * sigma);
    
    alpha_it = arma::sum( probs_h ) / arma::sum(probs_h_old_Ri) * gli_it(old_Ri_it) / gli_it(new_Ri_it) ;
    U_it     = arma::randu() ;
    
    if( alpha_it > U_it){
      
      // Accpeted transition
      Theta_list_ret[subject] = newTheta_it ; 
      Ri_vec(subject)         = new_Ri_it  ;
      rho(subject)            = new_rho ;
      
      AR += 1 ; 
      
      if(new_rho == H){
        rcpp_add_elem_1(gTable);
        new_xi_star = cpp_rdirichletArma1_plus_1_pos(a, new_Ri_it) ;
        rcpp_add_row(xi, new_xi_star) ;
        H += 1 ;
      }else{
        gTable(new_rho) += 1 ;
        }
    }else{ 
      
      Theta_list_ret[subject] = Theta_list_old[subject] ;
      rho(subject) = old_rho ; // needed for singletons
       
      if(old_rho == H){
        rcpp_add_elem_1(gTable);
        rcpp_add_row(xi, old_xi_star) ;
        H += 1 ;
      }else{ gTable(old_rho) += 1 ; } 
    } 
  } 
  
  Theta_list_ret.attr("names") = IDs_int ;
  
  return Rcpp::List::create(Rcpp::Named("up_AR")     = AR,
                            Rcpp::Named("up_Ri")     = Ri_vec,
                            Rcpp::Named("up_Gamma")  = Theta_list_ret,
                            Rcpp::Named("up_table")  = gTable, 
                            Rcpp::Named("up_xi")     = xi,
                            Rcpp::Named("up_rho")    = rho);
}  


//' Collapsed Gibbs sampler for the SUN-DPM model (full output)
//'
//' @description
//' Runs a collapsed Gibbs sampler for the model described in the paper. 
//'
//' For a memory-efficient variant that stores only
//' \eqn{(\boldsymbol{\delta}, \mathbf{R})} see
//' \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}}.
//'
//' @param data
//'   Named \code{Rcpp::List} of length \eqn{n}.  Each element is itself a
//'   list with at least a field \code{"ts"} (the binary time series for
//'   subject \eqn{i}) and any covariates needed by the update functions.
//' @param X
//'   \eqn{n \times q} matrix of subject-level static covariates
//'   \eqn{\mathbf{x}_i}.  Used to build the stacked probit design matrix
//'   and forwarded to the state-trajectory update.
//' @param IDs
//'   Character vector of length \eqn{n} with subject identifiers.
//' @param STATE
//'   Named \code{Rcpp::List} holding the current MCMC state.  Must contain
//'   at least the fields \code{"rho"}, \code{"delta"}, \code{"xi"},
//'   \code{"Gamma"}, and \code{"Ri"}.  Modified in place at each sweep.
//' @param Prior
//'   Named list of hyperparameters for the DP and probit blocks:
//'   \describe{
//'     \item{\code{"a"}}{Concentration vector \eqn{\mathbf{a}} of the
//'       base-measure Dirichlet prior on \eqn{\boldsymbol{\xi}^*}.}
//'     \item{\code{"M"}}{DP precision parameter \eqn{M > 0}.}
//'     \item{\code{"sigma"}}{Discount parameter \eqn{\sigma}.}
//'     \item{\code{"d0"}}{Prior mean vector for \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{"D0"}}{Diagonal of the prior precision matrix
//'       \eqn{\mathbf{D}_0} for \eqn{\boldsymbol{\delta}}.}
//'   }
//' @param gPrior
//'   Named list of state-space prior hyperparameters for the DLM prior
//'   \eqn{g}:
//'   \describe{
//'     \item{\code{"m0"}, \code{"S0"}}{Prior mean and covariance of the
//'       initial state \eqn{\mathbf{x}_0}.}
//'     \item{\code{"G1"}, \code{"G2"}}{Baseline and alternative-regime
//'       transition matrices \eqn{\mathbf{G}} and \eqn{\mathbf{G}^*}.}
//'     \item{\code{"Ve"}, \code{"Ves"}}{Baseline and alternative-regime
//'       innovation covariances
//'       \eqn{\boldsymbol{\Sigma}_\varepsilon} and
//'       \eqn{\boldsymbol{\Sigma}_\varepsilon^*}.}
//'     \item{\code{"gli"}}{\eqn{8 \times K} matrix mapping each risk
//'       category \eqn{R_i \in \{0,\ldots,7\}} to a cluster probability
//'       vector.}
//'   }
//' @param sample    Number of posterior draws to store.
//'   Default: \code{2000}.
//' @param thinning  Store one sample every \code{thinning} post-burn-in.
//'   Default: \code{10}.
//' @param burn      Number of burn-in.
//'   Default: \code{1000}.
//' @param nSim      Number of SMC particles forwarded to the trajectory
//'   sampler inside \code{rcpp_update_Ri_gamma_and_rho}.
//'   Default: \code{100}.
//'
//' @return A named \code{Rcpp::List} with five fields, each of length
//'   \code{sample}:
//'   \describe{
//'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
//'       of the probit coefficient vector \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{rho}}{\eqn{\texttt{sample} \times n} matrix of cluster
//'       assignment draws \eqn{\boldsymbol{\rho}}.}
//'     \item{\code{xi_star}}{List of \code{sample} draws of the cluster
//'       atom matrix \eqn{\boldsymbol{\xi}^*}.}
//'     \item{\code{Gamma}}{List of \code{sample} draws of the latent state
//'       trajectories \eqn{\boldsymbol{\Gamma}}.}
//'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
//'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
//'   }
//'
//' @seealso
//'   \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}},
//'   \code{\link{rcpp_get_Ri_wSplit}},
//'   \code{\link{rcpp_rCenteredLinearPredictor_RaoBlackwellAuxSMC}}
//'
//' @examples
//' \dontrun{
//' fit <- SUN_gibbs_collapsed_sampler_v2(
//'   data     = subject_list,
//'   X        = X_mat,
//'   IDs      = ids,
//'   STATE    = initial_state,
//'   Prior    = prior_list,
//'   gPrior   = g_prior_list,
//'   sample   = 2000L,
//'   thinning = 10L,
//'   burn     = 1000L,
//'   nSim     = 100L
//' )
// [[Rcpp::export]]
Rcpp::List SUN_gibbs_collapsed_sampler_v3( const Rcpp::List data,
                                           const arma::mat X,
                                           const Rcpp::CharacterVector IDs,
                                           Rcpp::List STATE,
                                           const Rcpp::List  Prior,
                                           const Rcpp::List gPrior,
                                           const int sample = 2000,
                                           const int thinning = 10,
                                           const int burn = 1000,
                                           const int nSim = 100){
  
  std::string msg = "Creating Constant for the MCMC...\n" ;
  Rcpp::Rcout << msg << std::flush ;
  
  const int n = IDs.size();
  
  const  arma::vec a = Prior["a"];
  double M = Prior["M"];
  double sigma = Prior["sigma"];
  arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
  arma::ivec gTable_int   = rcpp_arma_table(sg);
  
  const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
  const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
  const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
  const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]); 
  const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
  const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
  const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
  arma::colvec delta_it ; 
  
  const int q = X.n_cols ;       // x_i
  const int p = mu0_int.n_elem ; // z_itK
  
  arma::mat  return_rho(sample,  n);
  arma::mat  return_delta(sample, q) ;
  Rcpp::List return_xi_star(sample) ;
  Rcpp::List return_theta_list(sample) ;
  Rcpp::List return_Ri_list(sample) ;
  
  Rcpp::List list_of_updates ;
  
  Rcpp::IntegerVector y_it ; 
  arma::vec xi_it ;
  arma::mat Zi_it ;
  Rcpp::IntegerVector Tstar_it ;
  double Ti_it ;  
  
  int AR_thetas ;
  int AR_delta ;
  
  // For Probit data augmentation 
  Rcpp::List Indeces_for_obs( n ) ;
  
  // DV 
  Rcpp::List SubjectData_intern;
  arma::vec y_intern ; 
  arma::rowvec xi_intern ;
  arma::uvec pos ;
   
  arma::mat    X_probit ;
  arma::colvec y_probit ;
  y_probit.set_size( 0 ) ; 
  
  const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
  const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
  
  for ( int subject = 0; subject < n; ++subject ) {
    
    SubjectData_intern  = data[ subject ] ;
     
    y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
    xi_intern = X.row(subject)  ;
    
    pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
    
    Indeces_for_obs[subject] = pos ;
    
    switch( subject ) {
    case 0:
      X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
      break;
    default: 
      X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
    }
    
    y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
  }  
  
  arma::colvec lower_limits( y_probit.n_elem ) ;
  arma::colvec upper_limits( y_probit.n_elem ) ;
  
  lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
  lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
  
  upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
  upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
  
  arma::mat invD0 = arma::inv( D0 ) ;
  arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
  
  
  // Gibbs
  
  int SampleStored = 0;
  const int MaxIteration = burn + sample * thinning ;
   
  // MCMC
  msg = "Running the chain...\n";
  Rcpp::Rcout << msg << std::endl;
  
  for (int it = 0; it < MaxIteration; ++it) {
    
    Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
    
    // MCMC CORE 
    // Theta {i=1, ... , N}
    list_of_updates = rcpp_update_Ri_gamma_and_rho_v3(
      data, STATE, gTable_int,
      IDs, a, M, sigma,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int,
      X, gl_matrix,
      AR_thetas) ; 
    
    STATE["Gamma"] = list_of_updates["up_Gamma"] ;
    STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
    STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
    STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
    gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
    AR_thetas      = list_of_updates["up_AR"]    ;
    
    // Xi moving
    STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
    
    // delta
    STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                      y_probit, X_probit, 
                                      lower_limits, upper_limits,
                                      Indeces_for_obs, 
                                      invD0, d0, V ) ;
    
    // STORE 
    if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
      return_rho.row(SampleStored)    = Rcpp::as<arma::rowvec>(STATE["rho"])   ;
      return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
      return_xi_star[SampleStored]    = STATE["xi"]    ;
      return_Ri_list[SampleStored]    = STATE["Ri"]    ;
      return_theta_list[SampleStored] = STATE["Gamma"] ;
      
      SampleStored = SampleStored + 1;
    }
    
    Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
    Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
    Rcpp::Rcout <<  gTable_int << std::flush;
  } 
  
  Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
  
  return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                             Rcpp::Named("rho")     = return_rho,
                             Rcpp::Named("xi_star") = return_xi_star,
                             Rcpp::Named("Gamma")   = return_theta_list,
                             Rcpp::Named("Ri")      = return_Ri_list );
} 



//' Collapsed Gibbs sampler for the SUN model (lightweight output)
//'
//' @description
//' Memory-efficient variant of
//' \code{\link{SUN_gibbs_collapsed_sampler_v2}}.  
//'
//' Use this variant when:
//' \itemize{
//'   \item only posterior inference on \eqn{\boldsymbol{\delta}} and the
//'     summaries statistics classification \eqn{p(R_i = r \mid \mathbf{y})} is needed;
//'   \item the number of subjects \eqn{n} or series length \eqn{T} makes
//'     storing full trajectories prohibitive.
//' }
//'
//' All parameters are identical to
//' \code{\link{SUN_gibbs_collapsed_sampler_v2}}; see that page for full
//' descriptions.
//'
//' @inheritParams SUN_gibbs_collapsed_sampler_v2
//'
//' @return A named \code{Rcpp::List} with two fields:
//'   \describe{
//'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
//'       of \eqn{\boldsymbol{\delta}}.}
//'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
//'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
//'   }
//'
//' @seealso \code{\link{SUN_gibbs_collapsed_sampler_v2}}
//'
//' @examples
//' \dontrun{
//' }
// [[Rcpp::export]]
Rcpp::List SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta_v3( const Rcpp::List data,
                                                              const arma::mat X,
                                                              const Rcpp::CharacterVector IDs,
                                                              Rcpp::List STATE,
                                                              const Rcpp::List  Prior,
                                                              const Rcpp::List gPrior,
                                                              const int sample = 2000,
                                                              const int thinning = 10,
                                                              const int burn = 1000,
                                                              const int nSim = 100){
  
  std::string msg = "Creating Constant for the MCMC...\n" ;
  Rcpp::Rcout << msg << std::flush ;
  
  const int n = IDs.size();
  
  const  arma::vec a = Prior["a"];
  double M = Prior["M"];
  double sigma = Prior["sigma"];
  arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
  arma::ivec gTable_int   = rcpp_arma_table(sg);
  
  const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
  const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
  const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
  const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]);  
  const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
  const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
  const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
  arma::colvec delta_it ; 
  
  const int q = X.n_cols ;       // x_i
  const int p = mu0_int.n_elem ; // z_itK
  
  arma::mat  return_delta(sample, q) ;
  Rcpp::List return_Ri_list(sample) ;
  
  Rcpp::List list_of_updates ;
  
  Rcpp::IntegerVector y_it ; 
  arma::vec xi_it ;
  arma::mat Zi_it ;
  Rcpp::IntegerVector Tstar_it ;
  double Ti_it ;  
  
  int AR_thetas ;
  int AR_delta ;
  
  // For Probit data augmentation 
  Rcpp::List Indeces_for_obs( n ) ;
  
  // DV 
  Rcpp::List SubjectData_intern;
  arma::vec y_intern ; 
  arma::rowvec xi_intern ;
  arma::uvec pos ;
  
  arma::mat    X_probit ;
  arma::colvec y_probit ;
  y_probit.set_size( 0 ) ; 
  
  const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
  const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
  
  for ( int subject = 0; subject < n; ++subject ) {
    
    SubjectData_intern  = data[ subject ] ;
    
    y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
    xi_intern = X.row(subject)  ;
    
    pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
    
    Indeces_for_obs[subject] = pos ;
    
    switch( subject ) {
    case 0:
      X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
      break;
    default:
      X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
    }
    
    y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
  }
  
  arma::colvec lower_limits( y_probit.n_elem ) ;
  arma::colvec upper_limits( y_probit.n_elem ) ;
  
  lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
  lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
  
  upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
  upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
  
  arma::mat invD0 = arma::inv( D0 ) ;
  arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
  
  
  // Gibbs
  
  int SampleStored = 0;
  const int MaxIteration = burn + sample * thinning ;
  
  // MCMC
  msg = "Running the chain...\n";
  Rcpp::Rcout << msg << std::endl;
  
  for (int it = 0; it < MaxIteration; ++it) {
    
    Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
    
    // MCMC CORE 
    
    // Theta {i=1, ... , N}
    list_of_updates = rcpp_update_Ri_gamma_and_rho_v3(
      data, STATE, gTable_int,
      IDs, a, M, sigma,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int,
      X, gl_matrix,
      AR_thetas) ;
    
    STATE["Gamma"] = list_of_updates["up_Gamma"] ;
    STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
    STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
    STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
    gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
    AR_thetas      = list_of_updates["up_AR"]    ;
    
    // Xi moving
    STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
    
    // delta
    STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                      y_probit, X_probit, 
                                      lower_limits, upper_limits,
                                      Indeces_for_obs, 
                                      invD0, d0, V ) ;
    
    // STORE
    if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
      return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
      return_Ri_list[SampleStored]    = STATE["Ri"]    ;
      
      SampleStored = SampleStored + 1;
    }
    // Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
    // Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
    // Rcpp::Rcout <<  gTable_int << std::flush;
  } 
  
  Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
  
  return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                             Rcpp::Named("Ri")      = return_Ri_list );
}  


// ============================================================
//  Revision functions: S2 < 2716 < S3 
// ============================================================

// [[Rcpp::export]]
int rcpp_get_Ri_wSplit_v4(const arma::mat & theta,
                          const arma::mat & Zi,
                          const arma::vec & Tstar){
  
  arma::vec eta = arma::sum(theta % Zi,1) ;
  // R1
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count  = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    } 
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    }  
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count )-1 ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
  }  
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int T_starPost ;
  
  if( Tstar.n_elem == 1){
    T_starPost = 0;
  }else{ 
    T_starPost = Tstar(Tstar.n_elem-2) ;
  } 

  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  
  if ( strRi == "00") {
    return 0;
  } else if ( strRi == "10") { 
    return 1;
  } else if ( strRi == "01") { 
    return 2;
  } else { 
    return 3;
  }
}

// [[Rcpp::export]]
int rcpp_get_Ri_wSplit_from_eta_v4(const arma::vec eta, 
                                      const arma::vec Tstar){
  
  // R1
  double avg = arma::mean(eta) ;
  int Ti = eta.n_elem ;
  
  int days_at_high_risk = 0;
  int days_at_risk = 0 ; 
  int Tstar_count  = 0  ; 
  double starting_int = eta(0) ;
  
  for (int i = 0; i < Ti; ++i) {
    
    if ( (eta(i) ) >  1) {
      days_at_risk++;
    } 
    
    if ( (eta(i) ) >  1.64  ) {
      days_at_high_risk++;
    }  
    
    if( Tstar( Tstar_count ) == i ){
      starting_int = eta( Tstar( Tstar_count )-1 ) ;
      Tstar_count  = Tstar_count + 1 ;
    }
  }  
  
  double pR2 = days_at_high_risk *1.0 / std::max(days_at_risk,1) ;
  
  int T_starPost ;
  
  if( Tstar.n_elem == 1){
    T_starPost = 0;
  }else{ 
    T_starPost = Tstar(Tstar.n_elem-2) ;
  } 

  std::string strRi = "" ;
  
  strRi += avg   >  1    ? '1' : '0';
  strRi += pR2   > .5    ? '1' : '0';
  
  if ( strRi == "00") {
    return 0;
  } else if ( strRi == "01") { 
    return 1;
  } else if ( strRi == "10") { 
    return 2;
  } else { 
    return 3; 
  }
}



// [[Rcpp::export]]
arma::vec sample_Ri_under_g_wSplit_v4(
    const int TT,
    const arma::mat    & matFF,
    const arma::colvec & mu0,
    const arma::mat    & Sigma0,
    const arma::mat    & G,
    const arma::mat    & G_star, 
    const arma::mat    & SigmaEps,
    const arma::mat    & SigmaEps_star,
    const arma::vec    & Tstar,
    const int nSim = 10000 ){
  
  const int p = mu0.n_elem ;
  arma::vec theta_it(p);
  
  const arma::colvec zero_ex_ev = arma::zeros<arma::vec>(p);
  
  int star_count = 0;
  int star_times_minus_1 = Tstar.n_elem - 1;
  
  arma::mat Theta(TT, p) ;
  arma::vec Ris(nSim) ;
  
  for (int sim = 0; sim < nSim; ++sim){
    theta_it = mvrnormArma1( mu0, Sigma0) ;
    
    for (int tt = 0; tt < TT; ++tt){
      
      if(tt != Tstar(star_count)){
        theta_it = G * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps) ;
      }else{ 
        theta_it = G_star * theta_it + mvrnormArma1( zero_ex_ev, SigmaEps_star) ;      
        if(star_times_minus_1 != star_count) star_count = star_count + 1; 
      }
      Theta.row(tt) = theta_it.t() ;
    } 
    Ris(sim) = rcpp_get_Ri_wSplit_v4(Theta, matFF, Tstar) ;
  }
  return Ris ;
} 

Rcpp::List rcpp_update_Ri_gamma_and_rho_v4(
    const Rcpp::List & data,
    const Rcpp::List & state, 
    arma::ivec & gTable,
    const Rcpp::CharacterVector & IDs_int,
    const arma::vec & a,
    const double M,
    const double sigma,
    const arma::vec &  mu0_int,
    const arma::mat & Sigma0_int,
    const arma::mat & G_int,
    const arma::mat & G_star_int, 
    const arma::mat & SigmaEps_int,
    const arma::mat & SigmaEps_star_int,
    const arma::mat & X_int,
    const arma::mat & gl_matrix,
    int AR){
  
  // inizialization
  int n_subject = IDs_int.size() ;
  
  Rcpp::List Theta_list_old = state["Gamma"] ;
  Rcpp::List Theta_list_ret(n_subject); 
  Rcpp::List SubjectData ;
  
  const arma::colvec delta_int = state["delta"] ;
  arma::ivec rho = state["rho"] ;
  arma::mat  xi  = state["xi"]  ;
  arma::colvec Ri_vec = state["Ri"] ;
  
  int H = xi.n_rows ;
  
  int old_rho;
  int new_rho;
  
  arma::vec    Ex_new =  a / arma::sum( a );
  arma::colvec new_xi_star ;
  arma::colvec old_xi_star ;
  arma::colvec rho_probs ;
  
  arma::colvec probs_h ;
  arma::colvec probs_h_old_Ri ;
  arma::uvec   labels ;
  
  // Place Holders for subjects
  Rcpp::IntegerVector ts_it ;
  arma::colvec        Tstar_it ;
  arma::mat           Zi_it ;
  arma::colvec        x_it ;
  
  int old_Ri_it ;
  int new_Ri_it ;
  
  double alpha_it ; 
  double U_it; 
  
  arma::vec gli_it;
  arma::mat newTheta_it ;
  
  // Subject Loop 
  for (int subject = 0; subject < n_subject; ++subject){
    
    // load r.v. for the iteration
    SubjectData = data[ subject ] ;
    ts_it       = Rcpp::as<Rcpp::IntegerVector>(SubjectData["ts"]) ;
    Zi_it       = Rcpp::as<arma::mat>(SubjectData["Z"]) ;
    Tstar_it    = Rcpp::as<arma::colvec>(SubjectData["Tstar"]) ;
    x_it        = X_int.row( subject ).t() ;
    gli_it      = gl_matrix.row(subject).t() ;
    
    // old Ri= rl and old rho = h
    old_Ri_it = Ri_vec(subject);
    old_rho   = rho(subject);
    
    //new Ri= rl' from g(\\theta_i)
    newTheta_it = rcpp_rEta_conditioned_RaoBlackwellAuxSMC(
      ts_it, delta_int , x_it, Zi_it,
      mu0_int, Sigma0_int,
      G_int, G_star_int,
      SigmaEps_int, SigmaEps_star_int , 
      SubjectData["Tstar"]);
    new_Ri_it = rcpp_get_Ri_wSplit_from_eta_v4(newTheta_it, Tstar_it);
    
    // remove the rho \xi* and \rho from the State
    if( gTable(old_rho) == 1){
      old_xi_star  = xi.row(old_rho).t();
      gTable       = rcpp_remove_element( gTable, old_rho) ;
      xi           = rcpp_remove_row( xi, old_rho) ;
      rho          = rcpp_correct_labels( rho, old_rho) ;
      
      H            -= 1 ;
      old_rho      = H  ; 
    }else{  
      gTable(old_rho) -= 1 ;
    }     
    
    labels = rcpp_index_gen( H+1 ) ;
    
    probs_h.resize(H + 1);
    probs_h.subvec(0,H-1)         = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(new_Ri_it) ; 
    probs_h(H)                    = Ex_new(new_Ri_it) * ( M + H * sigma);
    
    new_rho = rcpp_sample(labels,1, probs_h)(0);
    
    probs_h_old_Ri.resize(H+1) ; 
    probs_h_old_Ri.subvec(0,H-1)  = ( arma::conv_to<arma::vec>::from(gTable) - sigma) % xi.col(old_Ri_it) ; 
    probs_h_old_Ri(H)             = Ex_new(old_Ri_it) * ( M + H * sigma);
    
    alpha_it = arma::sum( probs_h ) / arma::sum(probs_h_old_Ri) * gli_it(old_Ri_it) / gli_it(new_Ri_it) ;
    U_it     = arma::randu() ;
    
    if( alpha_it > U_it){
      
      // Accpeted transition
      Theta_list_ret[subject] = newTheta_it ; 
      Ri_vec(subject)         = new_Ri_it  ;
      rho(subject)            = new_rho ;
      
      AR += 1 ; 
      
      if(new_rho == H){
        rcpp_add_elem_1(gTable);
        new_xi_star = cpp_rdirichletArma1_plus_1_pos(a, new_Ri_it) ;
        rcpp_add_row(xi, new_xi_star) ;
        H += 1 ;
      }else{
        gTable(new_rho) += 1 ;
      }
    }else{ 
      
      Theta_list_ret[subject] = Theta_list_old[subject] ;
      rho(subject) = old_rho ; // needed for singletons
      
      if(old_rho == H){
        rcpp_add_elem_1(gTable);
        rcpp_add_row(xi, old_xi_star) ;
        H += 1 ;
      }else{ gTable(old_rho) += 1 ; } 
    } 
  } 
  
  Theta_list_ret.attr("names") = IDs_int ;
  
  return Rcpp::List::create(Rcpp::Named("up_AR")     = AR,
                            Rcpp::Named("up_Ri")     = Ri_vec,
                            Rcpp::Named("up_Gamma")  = Theta_list_ret,
                            Rcpp::Named("up_table")  = gTable, 
                            Rcpp::Named("up_xi")     = xi,
                            Rcpp::Named("up_rho")    = rho);
}  


//' Collapsed Gibbs sampler for the SUN-DPM model (full output)
 //'
 //' @description
 //' Runs a collapsed Gibbs sampler for the model described in the paper. 
 //'
 //' For a memory-efficient variant that stores only
 //' \eqn{(\boldsymbol{\delta}, \mathbf{R})} see
 //' \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}}.
 //'
 //' @param data
 //'   Named \code{Rcpp::List} of length \eqn{n}.  Each element is itself a
 //'   list with at least a field \code{"ts"} (the binary time series for
 //'   subject \eqn{i}) and any covariates needed by the update functions.
 //' @param X
 //'   \eqn{n \times q} matrix of subject-level static covariates
 //'   \eqn{\mathbf{x}_i}.  Used to build the stacked probit design matrix
 //'   and forwarded to the state-trajectory update.
 //' @param IDs
 //'   Character vector of length \eqn{n} with subject identifiers.
 //' @param STATE
 //'   Named \code{Rcpp::List} holding the current MCMC state.  Must contain
 //'   at least the fields \code{"rho"}, \code{"delta"}, \code{"xi"},
 //'   \code{"Gamma"}, and \code{"Ri"}.  Modified in place at each sweep.
 //' @param Prior
 //'   Named list of hyperparameters for the DP and probit blocks:
 //'   \describe{
 //'     \item{\code{"a"}}{Concentration vector \eqn{\mathbf{a}} of the
 //'       base-measure Dirichlet prior on \eqn{\boldsymbol{\xi}^*}.}
 //'     \item{\code{"M"}}{DP precision parameter \eqn{M > 0}.}
 //'     \item{\code{"sigma"}}{Discount parameter \eqn{\sigma}.}
 //'     \item{\code{"d0"}}{Prior mean vector for \eqn{\boldsymbol{\delta}}.}
 //'     \item{\code{"D0"}}{Diagonal of the prior precision matrix
 //'       \eqn{\mathbf{D}_0} for \eqn{\boldsymbol{\delta}}.}
 //'   }
 //' @param gPrior
 //'   Named list of state-space prior hyperparameters for the DLM prior
 //'   \eqn{g}:
 //'   \describe{
 //'     \item{\code{"m0"}, \code{"S0"}}{Prior mean and covariance of the
 //'       initial state \eqn{\mathbf{x}_0}.}
 //'     \item{\code{"G1"}, \code{"G2"}}{Baseline and alternative-regime
 //'       transition matrices \eqn{\mathbf{G}} and \eqn{\mathbf{G}^*}.}
 //'     \item{\code{"Ve"}, \code{"Ves"}}{Baseline and alternative-regime
 //'       innovation covariances
 //'       \eqn{\boldsymbol{\Sigma}_\varepsilon} and
 //'       \eqn{\boldsymbol{\Sigma}_\varepsilon^*}.}
 //'     \item{\code{"gli"}}{\eqn{8 \times K} matrix mapping each risk
 //'       category \eqn{R_i \in \{0,\ldots,7\}} to a cluster probability
 //'       vector.}
 //'   }
 //' @param sample    Number of posterior draws to store.
 //'   Default: \code{2000}.
 //' @param thinning  Store one sample every \code{thinning} post-burn-in.
 //'   Default: \code{10}.
 //' @param burn      Number of burn-in.
 //'   Default: \code{1000}.
 //' @param nSim      Number of SMC particles forwarded to the trajectory
 //'   sampler inside \code{rcpp_update_Ri_gamma_and_rho}.
 //'   Default: \code{100}.
 //'
 //' @return A named \code{Rcpp::List} with five fields, each of length
 //'   \code{sample}:
 //'   \describe{
 //'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
 //'       of the probit coefficient vector \eqn{\boldsymbol{\delta}}.}
 //'     \item{\code{rho}}{\eqn{\texttt{sample} \times n} matrix of cluster
 //'       assignment draws \eqn{\boldsymbol{\rho}}.}
 //'     \item{\code{xi_star}}{List of \code{sample} draws of the cluster
 //'       atom matrix \eqn{\boldsymbol{\xi}^*}.}
 //'     \item{\code{Gamma}}{List of \code{sample} draws of the latent state
 //'       trajectories \eqn{\boldsymbol{\Gamma}}.}
 //'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
 //'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
 //'   }
 //'
 //' @seealso
 //'   \code{\link{SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta}},
 //'   \code{\link{rcpp_get_Ri_wSplit}},
 //'   \code{\link{rcpp_rCenteredLinearPredictor_RaoBlackwellAuxSMC}}
 //'
 //' @examples
 //' \dontrun{
 //' fit <- SUN_gibbs_collapsed_sampler_v2(
 //'   data     = subject_list,
 //'   X        = X_mat,
 //'   IDs      = ids,
 //'   STATE    = initial_state,
 //'   Prior    = prior_list,
 //'   gPrior   = g_prior_list,
 //'   sample   = 2000L,
 //'   thinning = 10L,
 //'   burn     = 1000L,
 //'   nSim     = 100L
 //' )
 // [[Rcpp::export]]
 Rcpp::List SUN_gibbs_collapsed_sampler_v4( const Rcpp::List data,
                                            const arma::mat X,
                                            const Rcpp::CharacterVector IDs,
                                            Rcpp::List STATE,
                                            const Rcpp::List  Prior,
                                            const Rcpp::List gPrior,
                                            const int sample = 2000,
                                            const int thinning = 10,
                                            const int burn = 1000,
                                            const int nSim = 100){
   
   std::string msg = "Creating Constant for the MCMC...\n" ;
   Rcpp::Rcout << msg << std::flush ;
   
   const int n = IDs.size();
   
   const  arma::vec a = Prior["a"];
   double M = Prior["M"];
   double sigma = Prior["sigma"];
   arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
   arma::ivec gTable_int   = rcpp_arma_table(sg);
   
   const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
   const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
   const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
   const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]); 
   const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
   const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
   const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
   arma::colvec delta_it ; 
   
   const int q = X.n_cols ;       // x_i
   const int p = mu0_int.n_elem ; // z_itK
   
   arma::mat  return_rho(sample,  n);
   arma::mat  return_delta(sample, q) ;
   Rcpp::List return_xi_star(sample) ;
   Rcpp::List return_theta_list(sample) ;
   Rcpp::List return_Ri_list(sample) ;
   
   Rcpp::List list_of_updates ;
   
   Rcpp::IntegerVector y_it ; 
   arma::vec xi_it ;
   arma::mat Zi_it ;
   Rcpp::IntegerVector Tstar_it ;
   double Ti_it ;  
   
   int AR_thetas ;
   int AR_delta ;
   
   // For Probit data augmentation 
   Rcpp::List Indeces_for_obs( n ) ;
   
   // DV 
   Rcpp::List SubjectData_intern;
   arma::vec y_intern ; 
   arma::rowvec xi_intern ;
   arma::uvec pos ;
   
   arma::mat    X_probit ;
   arma::colvec y_probit ;
   y_probit.set_size( 0 ) ; 
   
   const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
   const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
   
   for ( int subject = 0; subject < n; ++subject ) {
     
     SubjectData_intern  = data[ subject ] ;
     
     y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
     xi_intern = X.row(subject)  ;
     
     pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
     
     Indeces_for_obs[subject] = pos ;
     
     switch( subject ) {
     case 0:
       X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
       break;
     default: 
       X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
     }
     
     y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
   }  
   
   arma::colvec lower_limits( y_probit.n_elem ) ;
   arma::colvec upper_limits( y_probit.n_elem ) ;
   
   lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
   lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
   
   upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
   upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
   
   arma::mat invD0 = arma::inv( D0 ) ;
   arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
   
   
   // Gibbs
   
   int SampleStored = 0;
   const int MaxIteration = burn + sample * thinning ;
   
   // MCMC
   msg = "Running the chain...\n";
   Rcpp::Rcout << msg << std::endl;
   
   for (int it = 0; it < MaxIteration; ++it) {
     
     Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
     
     // MCMC CORE 
     // Theta {i=1, ... , N}
     list_of_updates = rcpp_update_Ri_gamma_and_rho_v4(
       data, STATE, gTable_int,
       IDs, a, M, sigma,
       mu0_int, Sigma0_int,
       G_int, G_star_int,
       SigmaEps_int, SigmaEps_star_int,
       X, gl_matrix,
       AR_thetas) ; 
     
     STATE["Gamma"] = list_of_updates["up_Gamma"] ;
     STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
     STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
     STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
     gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
     AR_thetas      = list_of_updates["up_AR"]    ;
     
     // Xi moving
     STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
     
     // delta
     STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                       y_probit, X_probit, 
                                       lower_limits, upper_limits,
                                       Indeces_for_obs, 
                                       invD0, d0, V ) ;
     
     // STORE 
     if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
       return_rho.row(SampleStored)    = Rcpp::as<arma::rowvec>(STATE["rho"])   ;
       return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
       return_xi_star[SampleStored]    = STATE["xi"]    ;
       return_Ri_list[SampleStored]    = STATE["Ri"]    ;
       return_theta_list[SampleStored] = STATE["Gamma"] ;
       
       SampleStored = SampleStored + 1;
     }
     
     Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
     Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
     Rcpp::Rcout <<  gTable_int << std::flush;
   } 
   
   Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
   
   return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                              Rcpp::Named("rho")     = return_rho,
                              Rcpp::Named("xi_star") = return_xi_star,
                              Rcpp::Named("Gamma")   = return_theta_list,
                              Rcpp::Named("Ri")      = return_Ri_list );
 } 



//' Collapsed Gibbs sampler for the SUN model (lightweight output)
 //'
 //' @description
 //' Memory-efficient variant of
 //' \code{\link{SUN_gibbs_collapsed_sampler_v2}}.  
 //'
 //' Use this variant when:
 //' \itemize{
 //'   \item only posterior inference on \eqn{\boldsymbol{\delta}} and the
 //'     summaries statistics classification \eqn{p(R_i = r \mid \mathbf{y})} is needed;
 //'   \item the number of subjects \eqn{n} or series length \eqn{T} makes
 //'     storing full trajectories prohibitive.
 //' }
 //'
 //' All parameters are identical to
 //' \code{\link{SUN_gibbs_collapsed_sampler_v2}}; see that page for full
 //' descriptions.
 //'
 //' @inheritParams SUN_gibbs_collapsed_sampler_v2
 //'
 //' @return A named \code{Rcpp::List} with two fields:
 //'   \describe{
 //'     \item{\code{delta}}{\eqn{\texttt{sample} \times q} matrix of draws
 //'       of \eqn{\boldsymbol{\delta}}.}
 //'     \item{\code{Ri}}{List of \code{sample} draws of the risk index
 //'       vector \eqn{\mathbf{R} = (R_1, \ldots, R_n)}.}
 //'   }
 //'
 //' @seealso \code{\link{SUN_gibbs_collapsed_sampler_v2}}
 //'
 //' @examples
 //' \dontrun{
 //' }
 // [[Rcpp::export]]
 Rcpp::List SUN_gibbs_collapsed_sampler_v2_Store_Ri_and_delta_v4( const Rcpp::List data,
                                                                  const arma::mat X,
                                                                  const Rcpp::CharacterVector IDs,
                                                                  Rcpp::List STATE,
                                                                  const Rcpp::List  Prior,
                                                                  const Rcpp::List gPrior,
                                                                  const int sample = 2000,
                                                                  const int thinning = 10,
                                                                  const int burn = 1000,
                                                                  const int nSim = 100){
   
   std::string msg = "Creating Constant for the MCMC...\n" ;
   Rcpp::Rcout << msg << std::flush ;
   
   const int n = IDs.size();
   
   const  arma::vec a = Prior["a"];
   double M = Prior["M"];
   double sigma = Prior["sigma"];
   arma::vec sg = Rcpp::as<arma::vec>( STATE["rho"] ) ;
   arma::ivec gTable_int   = rcpp_arma_table(sg);
   
   const arma::vec mu0_int            = Rcpp::as<arma::vec>(gPrior["m0"]);
   const arma::mat Sigma0_int         = Rcpp::as<arma::mat>(gPrior["S0"]);
   const arma::mat G_int              = Rcpp::as<arma::mat>(gPrior["G1"]);
   const arma::mat G_star_int         = Rcpp::as<arma::mat>(gPrior["G2"]);  
   const arma::mat SigmaEps_int       = Rcpp::as<arma::mat>(gPrior["Ve"]);
   const arma::mat SigmaEps_star_int  = Rcpp::as<arma::mat>(gPrior["Ves"]);
   const arma::mat gl_matrix          = Rcpp::as<arma::mat>(gPrior["gli"]);
   arma::colvec delta_it ; 
   
   const int q = X.n_cols ;       // x_i
   const int p = mu0_int.n_elem ; // z_itK
   
   arma::mat  return_delta(sample, q) ;
   Rcpp::List return_Ri_list(sample) ;
   
   Rcpp::List list_of_updates ;
   
   Rcpp::IntegerVector y_it ; 
   arma::vec xi_it ;
   arma::mat Zi_it ;
   Rcpp::IntegerVector Tstar_it ;
   double Ti_it ;  
   
   int AR_thetas ;
   int AR_delta ;
   
   // For Probit data augmentation 
   Rcpp::List Indeces_for_obs( n ) ;
   
   // DV 
   Rcpp::List SubjectData_intern;
   arma::vec y_intern ; 
   arma::rowvec xi_intern ;
   arma::uvec pos ;
   
   arma::mat    X_probit ;
   arma::colvec y_probit ;
   y_probit.set_size( 0 ) ; 
   
   const arma::vec d0 = Rcpp::as<arma::vec>( Prior["d0"] ) ;
   const arma::mat D0 = arma::diagmat( Rcpp::as<arma::vec>( Prior["D0"])) ;
   
   for ( int subject = 0; subject < n; ++subject ) {
     
     SubjectData_intern  = data[ subject ] ;
     
     y_intern = Rcpp::as<arma::vec>(SubjectData_intern["ts"]) ; 
     xi_intern = X.row(subject)  ;
     
     pos =  arma::find( (y_intern == 1 ) || (y_intern == 0) ) ;
     
     Indeces_for_obs[subject] = pos ;
     
     switch( subject ) {
     case 0:
       X_probit = arma::repmat( xi_intern, pos.n_elem, 1) ; 
       break;
     default:
       X_probit =  cpp_rbind( X_probit, arma::repmat( xi_intern, pos.n_elem, 1)); 
     }
     
     y_probit = arma::join_cols( y_probit, y_intern( pos ) ) ;
   }
   
   arma::colvec lower_limits( y_probit.n_elem ) ;
   arma::colvec upper_limits( y_probit.n_elem ) ;
   
   lower_limits.elem( arma::find( y_probit == 1 ) ).fill( 0.0 ) ;
   lower_limits.elem( arma::find( y_probit == 0 ) ).fill( R_NegInf ) ;
   
   upper_limits.elem( arma::find( y_probit == 1 ) ).fill( R_PosInf ) ;
   upper_limits.elem( arma::find( y_probit == 0 ) ).fill( 0.0 ) ;
   
   arma::mat invD0 = arma::inv( D0 ) ;
   arma::mat V = arma::inv( D0 + X_probit.t() * X_probit )  ; 
   
   
   // Gibbs
   
   int SampleStored = 0;
   const int MaxIteration = burn + sample * thinning ;
   
   // MCMC
   msg = "Running the chain...\n";
   Rcpp::Rcout << msg << std::endl;
   
   for (int it = 0; it < MaxIteration; ++it) {
     
     Rcpp::Rcout << "\r"  << " " << "[" << it << "/" << MaxIteration << "]            "  << std::flush;
     
     // MCMC CORE 
     
     // Theta {i=1, ... , N}
     list_of_updates = rcpp_update_Ri_gamma_and_rho_v4(
       data, STATE, gTable_int,
       IDs, a, M, sigma,
       mu0_int, Sigma0_int,
       G_int, G_star_int,
       SigmaEps_int, SigmaEps_star_int,
       X, gl_matrix,
       AR_thetas) ;
     
     STATE["Gamma"] = list_of_updates["up_Gamma"] ;
     STATE["Ri"]    = Rcpp::as<arma::colvec>(list_of_updates["up_Ri"])    ; 
     STATE["xi"]    = Rcpp::as<arma::mat>(list_of_updates["up_xi"]) ;
     STATE["rho"]   = Rcpp::as<arma::colvec>(list_of_updates["up_rho"]) ; 
     gTable_int     = Rcpp::as<arma::ivec>(list_of_updates["up_table"]) ;
     AR_thetas      = list_of_updates["up_AR"]    ;
     
     // Xi moving
     STATE["xi"] = rcpp_update_xi_star( STATE, a) ;
     
     // delta
     STATE["delta"] = rcpp_update_delta_v2( STATE, n,
                                       y_probit, X_probit, 
                                       lower_limits, upper_limits,
                                       Indeces_for_obs, 
                                       invD0, d0, V ) ;
     
     // STORE
     if ( it >= burn && ( (it - burn) % thinning == 0 ) ) {
       return_delta.row(SampleStored)  = Rcpp::as<arma::rowvec>(STATE["delta"]) ;
       return_Ri_list[SampleStored]    = STATE["Ri"]    ;
       
       SampleStored = SampleStored + 1;
     }
     // Rcpp::Rcout <<  Rcpp::as<arma::rowvec>(STATE["delta"]) << std::flush;
     // Rcpp::Rcout <<  Rcpp::as<arma::mat>(STATE["xi"]) << std::flush;
     // Rcpp::Rcout <<  gTable_int << std::flush;
   } 
   
   Rcpp::Rcout << "\r"   << " " << "[" << MaxIteration << "/" << MaxIteration << "]             "  << std::flush;
   
   return Rcpp::List::create( Rcpp::Named("delta")   = return_delta,
                              Rcpp::Named("Ri")      = return_Ri_list );
 }  
