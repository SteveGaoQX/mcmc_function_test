# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

dmvn_rcpp <- function(x, mean, sigma, logd = FALSE) {
    .Call(`_mcmcpackage_dmvn_rcpp`, x, mean, sigma, logd)
}

rmvn_rcpp <- function(n, mean, sigma) {
    .Call(`_mcmcpackage_rmvn_rcpp`, n, mean, sigma)
}

rinvgamma_rcpp <- function(a, b) {
    .Call(`_mcmcpackage_rinvgamma_rcpp`, a, b)
}

init_cpp <- function(Q, S, K, L, I, h_1, h_2) {
    .Call(`_mcmcpackage_init_cpp`, Q, S, K, L, I, h_1, h_2)
}

update_alpha_cpp <- function(beta, omega, sigma2, data_index, y, B, Z, g, Z_sum, V_alpha_inv, V_alpha_inv_mu_alpha, K, Q, S, I, J_max) {
    .Call(`_mcmcpackage_update_alpha_cpp`, beta, omega, sigma2, data_index, y, B, Z, g, Z_sum, V_alpha_inv, V_alpha_inv_mu_alpha, K, Q, S, I, J_max)
}

update_beta_cpp <- function(alpha, beta, omega, sigma2, B, Z, g, y, B_sum, B_sum_0, V_beta_inv, V_beta_inv_mu_beta, data_index, K, Q, L, I, J_max) {
    .Call(`_mcmcpackage_update_beta_cpp`, alpha, beta, omega, sigma2, B, Z, g, y, B_sum, B_sum_0, V_beta_inv, V_beta_inv_mu_beta, data_index, K, Q, L, I, J_max)
}

logpost_omega_cpp <- function(i, omega_i, alpha, beta, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max) {
    .Call(`_mcmcpackage_logpost_omega_cpp`, i, omega_i, alpha, beta, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max)
}

update_omega_cpp <- function(alpha, beta, omega, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max) {
    .Call(`_mcmcpackage_update_omega_cpp`, alpha, beta, omega, sigma2, Sigma_omega, y, Z, B, g, data_index, K, Q, J_max)
}

logpost_Sigma_omega_cpp <- function(Sigma_omega, omega) {
    .Call(`_mcmcpackage_logpost_Sigma_omega_cpp`, Sigma_omega, omega)
}

update_Sigma_omega_cpp <- function(omega, Sigma_omega) {
    .Call(`_mcmcpackage_update_Sigma_omega_cpp`, omega, Sigma_omega)
}

update_sigma2_cpp <- function(alpha, beta, omega, y, Z, B, data_index, g, h_1, h_2, K, J_max, I, Q) {
    .Call(`_mcmcpackage_update_sigma2_cpp`, alpha, beta, omega, y, Z, B, data_index, g, h_1, h_2, K, J_max, I, Q)
}

mcmc_update <- function(orig, data_index, y, B_cpp, Z_cpp, g_cpp, Z_sum, V_alpha_inv, V_alpha_inv_mu_alpha, K, Q, S, I, J_max, B_sum_cpp, B_sum_0, V_beta_inv, V_beta_inv_mu_beta, L, h_1, h_2) {
    .Call(`_mcmcpackage_mcmc_update`, orig, data_index, y, B_cpp, Z_cpp, g_cpp, Z_sum, V_alpha_inv, V_alpha_inv_mu_alpha, K, Q, S, I, J_max, B_sum_cpp, B_sum_0, V_beta_inv, V_beta_inv_mu_beta, L, h_1, h_2)
}

