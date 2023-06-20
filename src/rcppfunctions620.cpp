//[[Rcpp::depends(RcppArmadillo)]]
# include <RcppArmadillo.h>


using namespace std;
using namespace arma;
using namespace Rcpp;


const double log2pi = log(2.0*M_PI);


// [[Rcpp::export]]
double dmvn_rcpp(const arma::rowvec& x,const arma::rowvec& mean,const arma::mat& sigma, bool logd = false){
  //arma_rng::set_seed(123);
  // calculate density of multivariate normal distribution
  // args: x: row vector data
  //      mean: row vector mean, sigma: covariance matrix
  //      logd: true for taking log
  // returns: out: pdf (or log pdf) of multivariate normal distribution

  int xdim = x.size();
  arma::mat rooti = trans(inv(trimatu(chol(sigma))));
  double rootisum = sum(log(rooti.diag()));
  double constants = -(static_cast<double>(xdim)/2.0)*log2pi;

  arma::vec z = rooti*trans(x-mean);
  double out = constants-0.5*sum(z%z)+rootisum;

  if (logd == false){ out = exp(out); }
  return(out);
}



// [[Rcpp::export]]
arma::mat rmvn_rcpp(const int n,const arma::vec& mean,const arma::mat& sigma){
  //arma_rng::set_seed(123);
  // randomly generate samples from multivariate normal distribution
  // args: n: number of data
  //      mean: row vector mean, sigma: covariance matrix
  // returns: out: random samples from multivariate normal distribution

  int k = sigma.n_cols; // dimension of the multivariate normal distribution
  arma::mat z = randn(n, k);
  arma::mat out = repmat(mean,1,n).t()+z*chol(sigma);
  return(out);
}


// [[Rcpp::export]]
double rinvgamma_rcpp(double a, double b){

  // generate random samples from inverse-gamma distribution
  // args: inverse-gamma(a, b)
  // returns: random sample from inverse-gamma distribution

  return(1/R::rgamma(a, 1/b));
}



// [[Rcpp::export]]
Rcpp::List init_cpp(int Q, int S, int K, int L, int I, double h_1, double h_2){

  arma::vec mean(S, fill::zeros);
  arma::mat sigma = arma::eye(S, S);
  arma::mat alpha_init = rmvn_rcpp(Q, mean, sigma);

  //Rcpp::Rcout << "Mean size: " << mean.n_rows << "\n";
  //Rcpp::Rcout << "Sigma size: " << sigma.n_rows << "x" << sigma.n_cols << "\n";
  //Rcpp::Rcout << "Alpha_init size: " << alpha_init.n_rows << "x" << alpha_init.n_cols << "\n";
  arma::cube beta_init(K, Q, L);
  arma::vec beta_mean(L, fill::zeros);
  arma::mat beta_sigma = arma::eye(L, L);

  for (int k = 0; k < K; k++){
    arma::mat M = rmvn_rcpp(Q, beta_mean, beta_sigma);
    //Rcpp::Rcout << "Matrix M size: " << M.n_rows << "x" << M.n_cols << "\n";

    beta_init.subcube(k, 0, 0, k, Q-1, L-1) = M;
    //Rcpp::Rcout << "beta_init.subcube size: " << beta_init.subcube(k, 0, 0, k, Q-1, L-1).n_cols << "x" << beta_init.subcube(k, 0, 0, k, Q-1, L-1).n_slices << "\n";
  }

  arma::mat Sigma_omega_init = arma::eye(Q, Q);
  arma::mat omega_init(I, Q);
  mean.set_size(Q);
  for (int i = 0; i < I; i++){
    omega_init.row(i) = rmvn_rcpp(1, mean, Sigma_omega_init).row(0);
  }

  arma::vec sigma2_init(Q);
  for (int q = 0; q < Q; q++){
    sigma2_init(q) = rinvgamma_rcpp(h_1, h_2);
  }

  return Rcpp::List::create(Rcpp::Named("alpha") = alpha_init,
                            Rcpp::Named("beta") = beta_init,
                            Rcpp::Named("Sigma_omega") = Sigma_omega_init,
                            Rcpp::Named("omega") = omega_init,
                            Rcpp::Named("sigma2") = Rcpp::wrap(sigma2_init));
}


// [[Rcpp::export]]
arma::mat update_alpha_cpp(const arma::cube& beta, const arma::mat& omega, const arma::vec& sigma2,
                            const arma::cube& data_index, const arma::cube& y, const arma::field<arma::cube>& B,
                            const arma::field<arma::cube>& Z, const arma::vec& g, const arma::cube& Z_sum,
                            const arma::mat& V_alpha_inv, const arma::vec& V_alpha_inv_mu_alpha, int K, int Q, int S, int I, int J_max){
  // Update the alpha
  // args: 1: beta is K*Q*L at t-1 time
  //       2: omega is the I*Q mat at t-1 time
  //       3: sigma2 is the length Q col vect at t-1 time
  // returns: the updates alpha

  arma::mat alpha_update(Q, S, arma::fill::none);
  arma::mat V_n(S, S);
  arma::vec mu_n(S);

  for (int q = 0; q < Q; ++q){

    arma::vec Zy_sum(S, arma::fill::zeros);

    for (int i = 0; i < I; i++){
      //Rcpp::Rcout << "Processing for q: " << q + 1 << ", i: "<< i +1<< std::endl;

      int k = g[i] ;
      arma::uvec data_index_iq = arma::find(data_index.tube(i, q) == 1);

      //Rcpp::Rcout << "Size of data_index_iq: " << data_index_iq.size() << std::endl;
      //Rcpp::Rcout << "data_index_iq: " << data_index_iq << std::endl;

      arma::vec omega_vec(data_index_iq.size(), arma::fill::zeros);
      omega_vec.fill(omega(i, q));

      //Rcpp::Rcout << "Size of omega_vec: " << omega_vec.size() << std::endl;
      //Rcpp::Rcout << "omega_vec: " << omega_vec << std::endl;

      arma::mat B_iq_raw = B(i).tube(q,0,q, J_max-1 );

      //Rcpp::Rcout << "Size of B_iq_raw: " << arma::size(B_iq_raw) << std::endl;
      //Rcpp::Rcout << "B_iq_raw: " << B_iq_raw << std::endl;

      arma::mat B_iq = B_iq_raw.rows(data_index_iq);
      //Rcpp::Rcout << "q : " << q  << " k : "<< k <<std::endl;

      arma::vec beta_kq = beta.tube(k, q);
      //Rcpp::Rcout << "Size of beta_kq: " << beta_kq.size() << std::endl;
      //Rcpp::Rcout << "beta_kq: " << beta_kq << std::endl;
      arma::vec beta_Kq = beta.tube(K-1, q);



      //Rcpp::Rcout << "Size of beta_Kq: " << beta_Kq.size() << std::endl;
      //Rcpp::Rcout << "beta_Kq: " << beta_Kq << std::endl;

      arma::vec y_vec = arma::vectorise(y.tube(i, q));
      arma::vec y_selected = y_vec.elem(data_index_iq);

      //Rcpp::Rcout << "Size of y_vec: " << y_selected.size() << std::endl;
      //Rcpp::Rcout << "y_vec: " << y_vec << std::endl;

      //Rcpp::Rcout << "Size of y_selected: " << y_selected.size() << std::endl;
      //Rcpp::Rcout << "y_sel: " << y_selected << std::endl;

      arma::vec y_tilde_iq;
      if (k == K-1){ // baseline
        y_tilde_iq = y_selected - B_iq * beta_kq - omega_vec;
      } else { // not baseline
        y_tilde_iq = y_selected - B_iq * beta_Kq -
          B_iq * beta_kq - omega_vec;
      }
      //Rcpp::Rcout << "Size of y_tilde: " << y_tilde_iq.size() << std::endl;
      //Rcpp::Rcout << "y_tilde: " << y_tilde_iq << std::endl;


      arma::mat Z_iq_raw = Z(i).tube(q,0,q, J_max-1);
      //Rcpp::Rcout << "Size of Z_iq_raw: " << Z_iq_raw.size() << std::endl;
      //Rcpp::Rcout << "Z_iq_raw: " << Z_iq_raw << std::endl;
      arma::mat Z_selected = Z_iq_raw.rows(data_index_iq);


      //Rcpp::Rcout << "Size of Z_selected: " << Z_selected.size() << std::endl;
      //Rcpp::Rcout << "Z_selected: " << Z_selected << std::endl;
      Zy_sum += Z_selected.t() * y_tilde_iq;
      // Print Zy_sum
      //Rcout << "Zy_sum:\n" << Zy_sum << std::endl;

      // Print the shape of Zy_sum
      //Rcout << "Length of Zy_sum: " << Zy_sum.size() << std::endl;
    }

    //Rcpp::Rcout << "now q =  " << q  << ", S = " << S << std::endl;

    arma::mat Z_sum_q = Z_sum.tube(q,0, q,S-1);
    V_n = arma::inv(Z_sum_q / sigma2(q) + V_alpha_inv);
    //Rcpp::Rcout << "V_n matrix for q: " << q + 1 << ": " << V_n << std::endl;

    mu_n = V_n * (Zy_sum / sigma2(q) + V_alpha_inv_mu_alpha);
    //Rcpp::Rcout << "mu_n vector for q: " << q + 1 << ": " << mu_n << std::endl;

    alpha_update.row(q) = rmvn_rcpp(1, mu_n, V_n).row(0);
  }
  return alpha_update;
}



// [[Rcpp::export]]
arma::cube update_beta_cpp(const arma::mat& alpha, const arma::cube& beta, const arma::mat& omega,
                       const arma::vec& sigma2,
                       const arma::field<arma::cube>& B, const arma::field<arma::cube>& Z,
                       const arma::vec& g, const arma::cube& y,
                       const arma::field<arma::cube>& B_sum, const arma::cube& B_sum_0,
                       const arma::mat& V_beta_inv, const arma::vec& V_beta_inv_mu_beta,
                       const arma::cube& data_index, int K, int Q, int L, int I, int J_max){
  // Update the beta
  // args: 1: alpha is Q*S at t time
  //       2: beta is K*Q*L at t-1 time
  //       3: omega is the I*Q mat at t-1 time
  //       4: sigma2 is the length Q col vect at t-1 time
  // returns: the updates alpha

  arma::cube beta_update(K, Q, L, arma::fill::none);
  arma::mat V_n(L, L);
  arma::vec mu_n(L);
  //Rcpp::Rcout << "Update beta_k k from 1 to K-1 " << std::endl ;

  for (int k = 0; k < K-1; ++k){
    for (int q = 0; q < Q; ++q){
      //Rcpp::Rcout << "This is k: " << k+1 << ", q: " << q+1 << "\n" << std::endl ;

      // pre-calculated values for posterior
      arma::vec By_sum(L, arma::fill::zeros);
      arma::uvec index_k = arma::find(g == k); // subject index in k-th class
      //Rcpp::Rcout << "index_k: " << index_k + 1 << std::endl;
      arma::vec beta_Kq = beta.tube(K-1, q); // this can be put outside the for loop for i

      for (auto i : index_k){ // i start from 0 to I - 1
        //Rcpp::Rcout << "i : " << i+1 << std::endl;
        arma::uvec data_index_iq = arma::find(data_index.tube(i, q) == 1); // index of data for i-th subject at q-th response
        arma::vec y_tilde_iq;

        arma::mat B_iq_raw = B(i).tube(q,0,q, J_max-1 );
        arma::mat B_iq = B_iq_raw.rows(data_index_iq);

        arma::mat Z_iq_raw = Z(i).tube(q,0,q, J_max-1 );
        arma::mat Z_iq = Z_iq_raw.rows(data_index_iq);


        arma::vec y_vec = arma::vectorise(y.tube(i, q));
        arma::vec y_selected = y_vec.elem(data_index_iq);

        arma::vec alpha_q = alpha.row(q).t();


        arma::vec omega_vec(data_index_iq.size(), arma::fill::zeros);
        omega_vec.fill(omega(i, q));


        y_tilde_iq = y_selected - Z_iq* alpha_q -  B_iq * beta_Kq - omega_vec;
        By_sum += B_iq.t() * y_tilde_iq;
      }
      //Rcpp::Rcout << "By_sum: " << "\n" << By_sum << "\n" << std::endl;

      // posterior mean and covariance matrix
      arma::cube B_sum_k = B_sum(k); // using field to get cube for kth class
      arma::mat B_sum_kq = B_sum_k.tube(q, 0, q, L-1); // B_sum_k is a QxLxL cube
      V_n = arma::inv_sympd(B_sum_kq / sigma2(q) + V_beta_inv);
      mu_n = V_n * (By_sum / sigma2(q) + V_beta_inv_mu_beta);
      //Rcpp::Rcout << "V_n: " << "\n" << V_n << "\n" << std::endl;

      //Rcpp::Rcout << "mu_n: " << "\n" << mu_n << "\n" << std::endl;


      beta_update.tube(k, q) = rmvn_rcpp(1, mu_n, V_n).row(0);

    }
  }


  //Rcpp::Rcout << "Update beta_0 " << std::endl ;

  // update beta_0
  for (int q = 0; q < Q; ++q){
    //Rcpp::Rcout << "Update K-1 for q :  "<< q+1 << std::endl ;

    // pre-calculated values for posterior
    arma::vec By_sum(L, arma::fill::zeros);
    for (int i = 0; i < I; ++i){
      int k = g[i]; // index of class for i-th subject
      arma::uvec data_index_iq = arma::find(data_index.tube(i, q) == 1); // index of data for i-th subject at q-th response
      arma::vec y_tilde_iq;

      arma::mat B_iq_raw = B(i).tube(q,0,q, J_max-1 );
      arma::mat B_iq = B_iq_raw.rows(data_index_iq);

      arma::mat Z_iq_raw = Z(i).tube(q,0,q, J_max-1 );
      arma::mat Z_iq = Z_iq_raw.rows(data_index_iq);

      //arma::vec beta_Kq = beta.tube(K-1, q); // this can be put outside the for loop for i

      arma::vec y_vec = arma::vectorise(y.tube(i, q));
      arma::vec y_selected = y_vec.elem(data_index_iq);

      arma::vec alpha_q = alpha.row(q).t();


      arma::vec omega_vec(data_index_iq.size(), arma::fill::zeros);
      omega_vec.fill(omega(i, q));



      if (k == K-1){
        // if i-th subject in baseline class
        y_tilde_iq = y_selected - Z_iq* alpha_q - omega_vec;
      } else {
        // if i-th subject not in baseline class
        arma::vec beta_kq = beta.tube(k, q);
        y_tilde_iq = y_selected - Z_iq* alpha_q - B_iq * beta_kq - omega_vec;
      }
      By_sum += B_iq.t() * y_tilde_iq;
    }

    // posterior mean and covariance matrix
    arma::mat B_sum_0_q = B_sum_0.tube(q, 0, q, L-1); // B_sum_0 is a QxLxL cube
    V_n = arma::inv_sympd(B_sum_0_q / sigma2(q) + V_beta_inv);
    mu_n = V_n * (By_sum / sigma2(q) + V_beta_inv_mu_beta);
    //Rcpp::Rcout << "V_n: " << "\n" << V_n << "\n" << std::endl;
    //Rcpp::Rcout << "mu_n: " << "\n" << mu_n << "\n" << std::endl;

    beta_update.tube(K-1, q) = rmvn_rcpp(1, mu_n, V_n).row(0);
  }

  return beta_update;
}

// [[Rcpp::export]]
double logpost_omega_cpp(int i, const arma::rowvec& omega_i, const arma::mat& alpha, const arma::cube& beta, const arma::vec& sigma2, const arma::mat& Sigma_omega,
                         const arma::cube& y, const arma::field<arma::cube>& Z, const arma::field<arma::cube>& B, const arma::vec& g,
                         const arma::cube& data_index, int K, int Q, int J_max){
  // calculate the log-posterior for omega_i
  // args: index of i-th subject, i from 0 to I-1 since it call by other Rcpp function
  //    omega_i: omega values for i-subjects, as a rowvector length Q
  double logpost = 0.0;

  // calculate log-likelihood
  int k = g(i) ; // index of class for i-th subject
  //Rcpp::Rcout << "k =  " << k << "\n";
  for (int q = 0; q < Q; ++q){
    //Rcpp::Rcout << "q =  " << q << "\n";
    arma::vec alpha_q = alpha.row(q).t();
    //Rcpp::Rcout << "alpha_q dimensions: " << alpha_q.n_rows << " x " << alpha_q.n_cols << "\n";

    arma::uvec data_index_iq = arma::find(data_index.tube(i, q) == 1); // index of data for i-th subject at q-th response
    //Rcpp::Rcout << "data_index_iq elements: " << data_index_iq.n_elem << "\n";

    arma::vec y_vec = arma::vectorise(y.tube(i, q));
    //Rcpp::Rcout << "y_vec dimensions: " << y_vec.n_rows << " x " << y_vec.n_cols << "\n";

    arma::vec y_selected = y_vec.elem(data_index_iq);
    //Rcpp::Rcout << "y_selected dimensions: " << y_selected.n_rows << " x " << y_selected.n_cols << "\n";

    arma::mat Z_iq_raw = Z(i).tube(q, 0, q, J_max-1);
    //Rcpp::Rcout << "Z_iq_raw dimensions: " << Z_iq_raw.n_rows << " x " << Z_iq_raw.n_cols << "\n";

    arma::mat Z_iq = Z_iq_raw.rows(data_index_iq);
    //Rcpp::Rcout << "Z_iq dimensions: " << Z_iq.n_rows << " x " << Z_iq.n_cols << "\n";

    arma::mat B_iq_raw = B(i).tube(q, 0, q, J_max-1);
    //Rcpp::Rcout << "B_iq_raw dimensions: " << B_iq_raw.n_rows << " x " << B_iq_raw.n_cols << "\n";

    arma::mat B_iq = B_iq_raw.rows(data_index_iq);
    //Rcpp::Rcout << "B_iq dimensions: " << B_iq.n_rows << " x " << B_iq.n_cols << "\n";
    arma::vec y_tilde_iq;

    arma::vec beta_Kq = beta.tube(K-1,q);
    arma::vec beta_kq;
    //Rcpp::Rcout << "beta_Kq dimensions: " << beta_Kq.n_rows << " x " << beta_Kq.n_cols << "\n";

    if(k != K-1){
      beta_kq = beta.tube(k,q);
      //Rcpp::Rcout << "beta_kq dimensions: " << beta_kq.n_rows << " x " << beta_kq.n_cols << "\n";

    }

    if (k == K - 1){
      // if i-th subject in baseline class
      y_tilde_iq = y_selected - Z_iq * alpha_q - B_iq * beta_Kq;
    } else {
      // if i-th subject not in baseline class
      y_tilde_iq = y_selected - Z_iq * alpha_q - B_iq * beta_Kq - B_iq * beta_kq;
    }
    //Rcpp::Rcout << "y_tilde_iq dimensions: " << y_tilde_iq.n_rows << " x " << y_tilde_iq.n_cols << "\n";
    arma::rowvec y_tilde_iq_rowvec = y_tilde_iq.t();

    int J_iq = data_index_iq.size();
    //Rcpp::Rcout << "J_iq:" << J_iq << std::endl;

    arma::vec mu(J_iq, arma::fill::ones);
    //Rcpp::Rcout << "omega_ilength:" << omega_i.n_cols << std::endl;

    mu *= omega_i(q);
    //Rcpp::Rcout << "mu:" << mu << std::endl;
    arma::rowvec mu_rowvec = mu.t();
    //Rcpp::Rcout << "mu_rowvec dimension: " << mu_rowvec.n_cols << std::endl;


    arma::mat sigma_mat = arma::eye<arma::mat>(J_iq, J_iq) * sigma2(q);
    //Rcpp::Rcout << "sigma_mat dimension: " << sigma_mat.n_rows << " x " << sigma_mat.n_cols << std::endl;

    // print sigma_mat content
    //Rcpp::Rcout << "sigma_mat content:\n" << sigma_mat << std::endl;

    logpost += dmvn_rcpp(y_tilde_iq_rowvec, mu_rowvec, sigma_mat, true);
  }
  //arma::rowvec omega_i_rowvec = omega_i.t();
  arma::rowvec zeros = arma::zeros<arma::rowvec>(Q);
  arma::rowvec omega_i_temp = omega_i;
  logpost += dmvn_rcpp(omega_i_temp, zeros, Sigma_omega, true);

  return logpost;
}



  // [[Rcpp::export]]
arma::mat update_omega_cpp(const arma::mat& alpha, const arma::cube& beta, const arma::mat& omega, const arma::vec& sigma2, const arma::mat& Sigma_omega,
                             const arma::cube& y, const arma::field<arma::cube>& Z, const arma::field<arma::cube>& B, const arma::vec& g,
                             const arma::cube& data_index, int K, int Q, int J_max){



  arma::mat omega_update = omega;
  double var_update = 0.01; // variance for normal random walk
  int I = omega.n_rows;

  for (int i = 0; i < I; i++){
    //Rcpp::Rcout << "i:  " << i+1 << "\n"<< std::endl;

    // propose new omega_i
    arma::rowvec omega_i_row = omega_update.row(i);
    arma::vec omega_i = arma::conv_to<arma::vec>::from(omega_i_row); // convert rowvec to vec
    arma::mat var_mat = arma::eye(Q,Q) * var_update;
    arma::mat omega_i_new = rmvn_rcpp(1, omega_i, var_mat);

    arma::rowvec omega_i_new_row = omega_i_new.row(0);

    // acceptance ratio
    double ratio = logpost_omega_cpp(i, omega_i_new_row, alpha, beta, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max) -
      logpost_omega_cpp(i, omega_i_row, alpha, beta, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max);
    //Rcpp::Rcout << "Ratio :  " << ratio << "\n" << std::endl;
    // accept or reject
    if (log(R::runif(0, 1)) < ratio){
      omega_update.row(i) = omega_i_new.row(0);
    }
  }

  return(omega_update);
}

// [[Rcpp::export]]
double logpost_Sigma_omega_cpp(const arma::mat& Sigma_omega, const arma::mat& omega){

  double logpost = 0.0;
  arma::rowvec zeros = arma::zeros<arma::rowvec>(omega.n_cols);
  int I = omega.n_rows; // Added a semicolon at the end

  for (int i = 0; i < I; i++){
    arma::rowvec omega_i = omega.row(i);
    logpost += dmvn_rcpp(omega_i, zeros, Sigma_omega, true);
  }

  return logpost;
}


// [[Rcpp::export]]
arma::mat update_Sigma_omega_cpp(const arma::mat& omega, const arma::mat& Sigma_omega){

  arma::mat Sigma_omega_update = Sigma_omega;
  double step = 0.03;
  double eps = 1e-300;
//int M = 300;

  int Q = Sigma_omega.n_rows;  // number of rows in Sigma_omega should match number of columns in omega
  //Rcpp::Rcout << "Q = "<< Q << std::endl;
  //for (int m = 0; m < M; m++){
    for (int i = 0; i < Q; i++){
      for (int j = 0; j < Q; j++){
      //Rcpp::Rcout << "now i : "<< i << "j:" << j << std::endl;

      if (i < j){
        double lower = std::max(Sigma_omega_update(i, j) - step, -1.0 + eps);
        double upper = std::min(Sigma_omega_update(i, j) + step, 1.0 - eps);

        arma::mat Sigma_omega_new = Sigma_omega_update;
        Sigma_omega_new(i, j) = R::runif(lower, upper);
        Sigma_omega_new(j, i) = Sigma_omega_new(i, j);

        if (Sigma_omega_new.is_sympd()){
          double ratio = logpost_Sigma_omega_cpp(Sigma_omega_new, omega) -
            logpost_Sigma_omega_cpp(Sigma_omega_update, omega);
          //Rcpp::Rcout << "update is pd"<< std::endl;

          if (std::log(R::runif(0, 1)) < ratio){
            Sigma_omega_update = Sigma_omega_new;
            //Rcpp::Rcout << "updata accpeted"<< std::endl;
          }
        }
      }
    }
  }

  return Sigma_omega_update;
}



// [[Rcpp::export]]
arma::vec update_sigma2_cpp(arma::mat& alpha, arma::cube& beta, arma::mat& omega,
                            const arma::cube& y, const arma::field<arma::cube>& Z, const arma::field<arma::cube>& B,
                            const arma::cube& data_index, const arma::vec& g, double h_1, double h_2, int K, int J_max, int I, int Q){
  arma::vec sigma2_update(Q, arma::fill::zeros);
  // Print beta dimensions
  //Rcpp::Rcout << "beta dimensions: " << beta.n_rows << " x " << beta.n_cols << " x " << beta.n_slices << "\n";

  // Print K
  //Rcpp::Rcout << "K: " << K << "\n";


  for (int q = 0; q < Q; ++q){
    //Rcpp::Rcout << "q : " << q << "\n";

    double y_tilde_sum = 0.0;

    arma::mat data_index_q = data_index.tube(0, q, I - 1, q);
    int n_iq = arma::accu(data_index_q);

    for (int i = 0; i < I; ++i){
      int k = g(i); // index of class for i-th subject
      //Rcpp::Rcout << "i : " << i << "\n";
      //Rcpp::Rcout << "k : " << k << "\n";
      arma::uvec data_index_iq = arma::find(data_index.tube(i, q) == 1); // index of data for i-th subject at q-th response
      //Rcpp::Rcout << "data_index_iq length: " << data_index_iq.n_elem << "\n";

      arma::mat B_iq_raw = B(i).tube(q, 0, q, J_max-1);
      //Rcpp::Rcout << "B_iq_raw dimensions: " << B_iq_raw.n_rows << " x " << B_iq_raw.n_cols << "\n";

      arma::mat B_iq = B_iq_raw.rows(data_index_iq);
      //Rcpp::Rcout << "B_iq dimensions: " << B_iq.n_rows << " x " << B_iq.n_cols << "\n";

      arma::mat Z_iq_raw = Z(i).tube(q, 0, q, J_max-1);
      //Rcpp::Rcout << "Z_iq_raw dimensions: " << Z_iq_raw.n_rows << " x " << Z_iq_raw.n_cols << "\n";

      arma::mat Z_iq = Z_iq_raw.rows(data_index_iq);
      //Rcpp::Rcout << "Z_iq dimensions: " << Z_iq.n_rows << " x " << Z_iq.n_cols << "\n";

      arma::vec y_vec = arma::vectorise(y.tube(i, q));
      //Rcpp::Rcout << "y_vec dimensions: " << y_vec.n_rows << " x " << y_vec.n_cols << "\n";

      arma::vec y_selected = y_vec.elem(data_index_iq);
      //Rcpp::Rcout << "y_selected dimensions: " << y_selected.n_rows << " x " << y_selected.n_cols << "\n";

      arma::vec alpha_q = alpha.row(q).t();
      //Rcpp::Rcout << "alpha_q dimensions: " << alpha_q.n_rows << " x " << alpha_q.n_cols << "\n";

      arma::vec omega_vec(data_index_iq.size(), arma::fill::zeros);
      omega_vec.fill(omega(i, q));
      //Rcpp::Rcout << "omega_vec dimensions: " << omega_vec.n_rows << " x " << omega_vec.n_cols << "\n";

      arma::vec beta_Kq = beta.tube(K-1, q);
      //Rcpp::Rcout << "beta_Kq dimensions: " << beta_Kq.n_rows << " x " << beta_Kq.n_cols << "\n";

      arma::vec y_tilde_iq = y_selected - Z_iq * alpha_q - B_iq * beta_Kq - omega_vec;
      if (k != K) {
        arma::vec beta_kq = beta.tube(k, q);
        //Rcpp::Rcout << "beta_kq dimensions: " << beta_kq.n_rows << " x " << beta_kq.n_cols << "\n";
        y_tilde_iq -= B_iq * beta_kq;
      }

      y_tilde_sum += arma::accu(arma::square(y_tilde_iq));
    }

    double h_1_star = h_1 + n_iq / 2;
    double h_2_star = h_2 + y_tilde_sum / 2;

    sigma2_update(q) = rinvgamma_rcpp(h_1_star, h_2_star);
  }

  return sigma2_update;
}




// [[Rcpp::export]]
List mcmc_update(const List& orig,
                 const arma::cube& data_index, const arma::cube& y, const arma::field<arma::cube>& B_cpp,
                 const arma::field<arma::cube>& Z_cpp, const arma::vec& g_cpp, const arma::cube& Z_sum,
                 const arma::mat& V_alpha_inv, const arma::vec& V_alpha_inv_mu_alpha,
                 const int K, const int Q, const int S, const int I, const int J_max,
                 const arma::field<arma::cube>& B_sum_cpp, const arma::cube& B_sum_0,
                 const arma::mat& V_beta_inv, const arma::vec& V_beta_inv_mu_beta,
                 const int L, const double h_1, const double h_2) {

  arma::cube beta_orig = as<arma::cube>(orig["beta"]);
  arma::mat omega_orig = as<arma::mat>(orig["omega"]);
  arma::vec sigma2_orig = as<arma::vec>(orig["sigma2"]);
  arma::mat Sigma_omega_orig = as<arma::mat>(orig["Sigma_omega"]);

  //Rcout << "Start update" << std::endl;
  arma::mat alpha_new = update_alpha_cpp(beta_orig,
                                         omega_orig,
                                         sigma2_orig,
                                          data_index, y, B_cpp, Z_cpp, g_cpp, Z_sum,
                                          V_alpha_inv, V_alpha_inv_mu_alpha,
                                          K, Q, S, I, J_max);
  //Rcout << "finish update alpha" << std::endl;
  arma::cube beta_new = update_beta_cpp(alpha_new,
                                        beta_orig,
                                        omega_orig,
                                        sigma2_orig,
                                        B_cpp, Z_cpp, g_cpp, y,
                                        B_sum_cpp, B_sum_0,
                                        V_beta_inv, V_beta_inv_mu_beta,
                                        data_index, K, Q, L, I, J_max);
  for (int i = 0; i < Q; i++) {
    beta_new.subcube(K-1, 0, 0, K-1, Q-1,0).fill(0.0); // baseline group zero intercept for identification
  }
  //Rcout << "finish update beta" << std::endl;

  arma::mat omega_new = update_omega_cpp(alpha_new, beta_new, omega_orig, sigma2_orig, Sigma_omega_orig, y, Z_cpp, B_cpp, g_cpp, data_index, K, Q, J_max);


  arma::mat Sigma_omega_new = update_Sigma_omega_cpp(omega_new, Sigma_omega_orig);

  arma::vec sigma2_new = update_sigma2_cpp(alpha_new, beta_new, omega_new,
                                           y, Z_cpp, B_cpp, data_index, g_cpp,
                                           h_1, h_2, K, J_max, I, Q);

  // Create output list
  List target = List::create(_["alpha"] = alpha_new,
                             _["beta"] = beta_new,
                             _["omega"] = omega_new,
                             _["Sigma_omega"] = Sigma_omega_new,
                             _["sigma2"] = sigma2_new);

  return target;
}














