#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
using namespace Rcpp;
using namespace arma;
using namespace std;

//' Backprop (Internal)
//'
//' @param n_hidden Hidden layer numbers.
//' @param w_ini Initial weights.
//' @param load_param Whether to load parameters.
//' @param weight Weights.
//' @param bias Biases.
//' @param x Training.
//' @param y Training.
//' @param w Training.
//' @param valid Whether to use a validation set.
//' @param x_valid Validation.
//' @param y_valid Validation.
//' @param w_valid Validation.
//' @param activ Activation function.
//' @param n_epoch Number of epochs.
//' @param n_batch Batch size.
//' @param model_type Model type.
//' @param learning_rate (Initial) learning rate.
//' @param l1_reg L1-penalty.
//' @param l2_reg L2-penalty.
//' @param early_stop Whether to early stop.
//' @param early_stop_det Number of epochs to determine early-stop.
//' @param learning_rate_adaptive Adaptive learning rate adjustment method.
//' @param rho Parameter.
//' @param epsilon Parameter.
//' @param beta1 Parameter.
//' @param beta2 Parameter.
//' @param loss_f Loss function.
//'
//' @return A list of outputs
//'
// [[Rcpp::export]]
SEXP backprop(NumericVector n_hidden, double w_ini,
              bool load_param, List weight, List bias,
              NumericMatrix x, NumericMatrix y, NumericVector w, bool valid,
              NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid,
              std::string activ,
              int n_epoch, int n_batch, std::string model_type,
              double learning_rate, double l1_reg, double l2_reg, bool early_stop, int early_stop_det,
              std::string learning_rate_adaptive, double rho, double epsilon, double beta1, double beta2,
              std::string loss_f) {

  mat x_ = as<mat>(x);
  mat y_ = as<mat>(y);
  vec w_ = as<vec>(w);
  mat x_valid_;
  mat y_valid_;
  vec w_valid_;

  if(valid) {

    x_valid_ = as<mat>(x_valid);
    y_valid_ = as<mat>(y_valid);
    w_valid_ = as<vec>(w_valid);
  }

  unsigned int sample_size = x.nrow();
  int n_layer = n_hidden.size();
  field<mat> weight_(n_layer + 1);
  field<vec> bias_(n_layer + 1);
  field<mat> a(n_layer + 1);
  field<mat> h(n_layer + 1);
  field<mat> d_a(n_layer + 1);
  field<mat> d_h(n_layer + 1);
  field<mat> d_w(n_layer + 1);
  vec loss(n_epoch);
  double best_loss = INFINITY;
  field<mat> best_weight(n_layer + 1);
  field<vec> best_bias(n_layer + 1);
  int break_k = n_epoch - 1;

  // adapative learning rate
  field<mat> dw(n_layer + 1);
  field<vec> db(n_layer + 1);
  // momentum
  field<mat> last_dw(n_layer + 1);
  field<vec> last_db(n_layer + 1);
  // adagrad
  field<mat> weight_ss(n_layer + 1);
  field<vec> bias_ss(n_layer + 1);
  // adadelta
  field<mat> weight_egs(n_layer + 1);
  field<vec> bias_egs(n_layer + 1);
  field<mat> weight_es(n_layer + 1);
  field<vec> bias_es(n_layer + 1);
  // adam
  field<mat> mt_w(n_layer + 1);
  field<vec> mt_b(n_layer + 1);
  field<mat> vt_w(n_layer + 1);
  field<vec> vt_b(n_layer + 1);
  // amsgrad
  field<mat> vt_bar_w(n_layer + 1);
  field<vec> vt_bar_b(n_layer + 1);
  double adam_ind = 0;
  double negbin_alpha = 1;
  double best_negbin_alpha = negbin_alpha;
  double log_negbin_alpha = log(negbin_alpha);
  double dalpha = 0;
  double last_dalpha = 0;
  double alpha_ss = 0;
  double alpha_egs = 0;
  double alpha_es = 0;
  double mt_alpha = 0;
  double vt_alpha = 0;
  double vt_bar_alpha = 0;

  for(int i = 0; i < n_layer + 1; i++) {

    if(i == 0) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(x_.n_cols, n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    } else if(i == n_layer) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], y_.n_cols) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(y_.n_cols) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
      db(i) = vec(y_.n_cols, fill::zeros);
    } else {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    }
  }

  best_weight = weight_;
  best_bias = bias_;

  if(learning_rate_adaptive == "momentum") {

    // MOMEMTUM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        last_dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        last_dw(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        last_db(i) = vec(y_.n_cols, fill::zeros);
      } else {
        last_dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adagrad") {

    // ADAGRAD
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_ss(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_ss(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_ss(i) = vec(y_.n_cols, fill::zeros);
      } else {
        weight_ss(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adadelta") {

    // ADADELTA
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_egs(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_egs(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_egs(i) = vec(y_.n_cols, fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_es(i) = vec(y_.n_cols, fill::zeros);
      } else {
        weight_egs(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adam") {

    // ADAM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        mt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_b(i) = vec(y_.n_cols, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "amsgrad") {

    // AMSGrad
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        mt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_bar_b(i) = vec(y_.n_cols, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  }

  int n_round = ceil((sample_size - 1)/n_batch);
  uvec i_bgn(n_round);
  uvec i_end(n_round);

  for(int s = 0; s < n_round; s++) {

    i_bgn[s] = s*n_batch;
    i_end[s] = (s+1)*n_batch - 1;
    if(i_end[s] > sample_size - 1) i_end[s] = sample_size - 1;
    if(s == n_round - 1) i_end[s] = sample_size - 1;
  }

  for(int k = 0; k < n_epoch; k++) {

    // shuffle
    uvec new_order = as<uvec>(sample(sample_size, sample_size)) - 1;
    x_ = x_.rows(new_order);
    y_ = y_.rows(new_order);
    w_ = w_.elem(new_order);

    for(int i = 0; i < n_round; i++) {

      mat xi_ = x_.rows(i_bgn[i], i_end[i]);
      mat yi_ = y_.rows(i_bgn[i], i_end[i]);
      vec wi_ = w_.subvec(i_bgn[i], i_end[i]);
      int n_s = xi_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);

      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {

          a(j) = xi_ * weight_(j) + one_sample_size * bias_(j).t();
        } else{

          a(j) = h(j-1) * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    h(j) = 1/(1+exp(-a(j)));
        else if(activ == "tanh")  h(j) = tanh(a(j));
        else if(activ == "relu")  h(j) = a(j) % (a(j) > 0);
        else if(activ == "prelu") h(j) = a(j) % (a(j) > 0) + (a(j) <= 0) % a(j)*0.2;
        else if(activ == "elu")   h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
        else if(activ == "celu")  h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
      }
      mat y_pi = h(n_layer - 1) * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      if(loss_f == "logit")
        y_pi = 1 / (1 + exp(-y_pi));

      if(loss_f == "rmsle") {

        y_pi = y_pi % (y_pi > 0);
        d_a(n_layer) = -(log(yi_ + 1) - log(y_pi + 1)) /
          (y_pi + 1) % y_pi.transform([](double x){return(1*(x > 0));}) % wi_ / sum(wi_);
      } else if(loss_f == "cross-entropy") {

        y_pi = exp(y_pi) / (sum(exp(y_pi), 1) * one_dim_y.t());
        d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else if(loss_f == "poisson") {

        d_a(n_layer) = -(yi_ - exp(y_pi)) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else if(loss_f == "negbin") {

        vec negvin_inv_plus_y = yi_ + 1/negbin_alpha;
        dalpha = -sum(sum(-(negvin_inv_plus_y.transform([](double x){return(R::digamma(x));}) -
          R::digamma(1/negbin_alpha))/negbin_alpha + log(1 + negbin_alpha*exp(y_pi))/negbin_alpha +
          (yi_ - exp(y_pi))/(1 + negbin_alpha*exp(y_pi))));
        d_a(n_layer) = -(yi_ - exp(y_pi)) / (1 + negbin_alpha*exp(y_pi)) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else if(loss_f == "poisson-nonzero") {

        d_a(n_layer) = -(yi_ - y_pi.transform([](double x){
          if(x < -20) {
            return(1.0);
          } else {
            return(exp(x)/(1-exp(-exp(x))));
          }})) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else if(loss_f == "negbin-nonzero") {

        vec lambda = exp(y_pi);
        vec one_plus_alpha_l = 1 + negbin_alpha*lambda;
        vec aaa = log(one_plus_alpha_l)/negbin_alpha;
        vec bbb = lambda/one_plus_alpha_l;
        vec negvin_inv_plus_y = yi_ + 1/negbin_alpha;
        dalpha = -sum(sum(-(negvin_inv_plus_y.transform([](double x){return(R::digamma(x));}) -
          R::digamma(1/negbin_alpha))/negbin_alpha + aaa + (yi_ - lambda)/one_plus_alpha_l -
          (-aaa + bbb) / (exp(aaa) - 1)));
        d_a(n_layer) = -((yi_ - lambda) / one_plus_alpha_l -
          bbb / (exp(aaa) - 1)) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else if(loss_f == "zip") {

        vec y_pi_0 = 1 / (1 + exp(-y_pi.col(0)));
        vec d_a_0 = -(1 - y_pi_0)/(1 - y_pi_0%(1-exp(-exp(y_pi.col(1))))) %
          (yi_.col(0) - y_pi_0%(1-exp(-exp(y_pi.col(1))))) % wi_ / sum(wi_);
        vec d_a_1 = -(yi_.col(0)%(yi_.col(1) - exp(y_pi.col(1))) -
          (1 - yi_.col(0))%y_pi_0%exp(y_pi.col(1) - exp(y_pi.col(1)))/
          (1 - y_pi_0%(1-exp(-exp(y_pi.col(1)))))) % wi_ / sum(wi_);
        d_a(n_layer) = join_horiz(d_a_0, d_a_1);
      } else if(loss_f == "zinb") {

        vec pi_ = 1 / (1 + exp(-y_pi.col(0)));
        vec lambda_ = exp(y_pi.col(1));
        vec one_plus_alpha_l = 1 + negbin_alpha*lambda_;
        vec aaa = log(one_plus_alpha_l)/negbin_alpha;
        vec bbb = lambda_/one_plus_alpha_l;
        vec ksi_ = 1 / (1 + exp(-(y_pi.col(0) - aaa)));
        vec d_a_0 = -pi_ + (1 - yi_.col(0)) % ksi_ + yi_.col(0);
        vec d_a_1 = (-(1 - yi_.col(0)) % ksi_ + yi_.col(0) % (yi_.col(1) / lambda_ - 1)) % bbb;
        vec y_plus_alpha_inv = yi_.col(1) + 1/negbin_alpha;
        d_a(n_layer) = join_horiz(-d_a_0 % wi_ / sum(wi_), -d_a_1 % wi_ / sum(wi_));
        dalpha = -sum(sum(-(1 - yi_.col(0)) % ksi_ % (-aaa + bbb) +
          yi_.col(0) % (-(y_plus_alpha_inv.transform([](double x){return(R::digamma(x));}) -
          R::digamma(1/negbin_alpha))/negbin_alpha + aaa + (yi_.col(1) / lambda_ - 1) % bbb)));
      } else {

        // d_a(n_layer) = -(yi_ - y_pi) % wi_ / sum(wi_);
        d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      }

      d_w(n_layer) = h(n_layer - 1).t() * d_a(n_layer);
      vec bias_grad = d_a(n_layer).t() * one_sample_size;
      if(learning_rate_adaptive == "momentum") {

        last_dw(n_layer) = last_dw(n_layer) * rho + d_w(n_layer) * learning_rate;
        last_db(n_layer) = last_db(n_layer) * rho + bias_grad    * learning_rate;
        dw(n_layer) = last_dw(n_layer);
        db(n_layer) = last_db(n_layer);
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          last_dalpha = last_dalpha * rho + dalpha * learning_rate;
          dalpha = last_dalpha;
        }
      } else if (learning_rate_adaptive == "adagrad") {

        weight_ss(n_layer) = weight_ss(n_layer) + square(d_w(n_layer));
        bias_ss(n_layer)   = bias_ss(n_layer)   + square(bias_grad);
        dw(n_layer) = d_w(n_layer)/sqrt(weight_ss(n_layer) + epsilon) * learning_rate;
        db(n_layer) = bias_grad   /sqrt(bias_ss(n_layer)   + epsilon) * learning_rate;
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          alpha_ss = alpha_ss + pow(dalpha, 2);
          dalpha = dalpha/sqrt(alpha_ss + epsilon) * learning_rate;
        }
      } else if (learning_rate_adaptive == "adadelta") {

        weight_egs(n_layer) = weight_egs(n_layer) * rho + (1-rho) * square(d_w(n_layer));
        bias_egs(n_layer)   = bias_egs(n_layer)   * rho + (1-rho) * square(bias_grad);
        dw(n_layer) = sqrt(weight_es(n_layer) + epsilon) / sqrt(weight_egs(n_layer) + epsilon) % d_w(n_layer);
        db(n_layer) = sqrt(bias_es(n_layer)   + epsilon) / sqrt(bias_egs(n_layer)   + epsilon) % bias_grad;
        weight_es(n_layer) = weight_es(n_layer) * rho + (1-rho) * square(dw(n_layer));
        bias_es(n_layer)   = bias_es(n_layer)   * rho + (1-rho) * square(db(n_layer));
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          alpha_egs = alpha_egs * rho + (1-rho) * pow(dalpha, 2);
          dalpha = sqrt(alpha_es + epsilon)/sqrt(alpha_egs + epsilon) * dalpha;
          alpha_es = alpha_es * rho + (1-rho) * pow(dalpha, 2);
        }
      } else if (learning_rate_adaptive == "adam") {

        adam_ind = adam_ind + 1;
        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        dw(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(n_layer) / (sqrt(vt_w(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
        db(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(n_layer) / (sqrt(vt_b(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          mt_alpha = mt_alpha * beta1 + (1-beta1) * dalpha;
          vt_alpha = vt_alpha * beta2 + (1-beta2) * pow(dalpha, 2);
          dalpha = learning_rate / (1-pow(beta1, adam_ind)) * mt_alpha / (sqrt(vt_alpha / (1-pow(beta2, adam_ind))) + epsilon);
        }
      } else if (learning_rate_adaptive == "amsgrad") {

        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        cube vt_bw(vt_w(n_layer).n_rows, vt_w(n_layer).n_cols, 2);
        mat  vt_bb(vt_b(n_layer).size(), 2);
        vt_bw.slice(0) = vt_w(n_layer);
        vt_bw.slice(1) = vt_bar_w(n_layer);
        vt_bb.col(0) = vt_b(n_layer);
        vt_bb.col(1) = vt_bar_b(n_layer);
        vt_bar_w(n_layer) = max(vt_bw, 2);
        vt_bar_b(n_layer) = max(vt_bb, 1);
        dw(n_layer) = learning_rate * mt_w(n_layer) / (sqrt(vt_bar_w(n_layer)) + epsilon);
        db(n_layer) = learning_rate * mt_b(n_layer) / (sqrt(vt_bar_b(n_layer)) + epsilon);
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          mt_alpha = mt_alpha * beta1 + (1-beta1) * dalpha;
          vt_alpha = vt_alpha * beta2 + (1-beta2) * pow(dalpha, 2);
          vec vt_balpha(2);
          vt_balpha(0) = vt_alpha;
          vt_balpha(1) = vt_bar_alpha;
          vt_bar_alpha = max(vt_balpha);
          dalpha = learning_rate * mt_alpha / (sqrt(vt_bar_alpha) + epsilon);
        }
      } else {

        dw(n_layer) = d_w(n_layer) * learning_rate;
        db(n_layer) = bias_grad    * learning_rate;
        if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
          dalpha = dalpha * learning_rate;
        }
      }
      weight_(n_layer) = weight_(n_layer) - dw(n_layer) - l1_reg * (conv_to<mat>::from(weight_(n_layer) > 0) - conv_to<mat>::from(weight_(n_layer) < 0)) - l2_reg * (weight_(n_layer));
      bias_(n_layer)   = bias_(n_layer)   - db(n_layer);
      if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
        log_negbin_alpha = log_negbin_alpha - dalpha;
        negbin_alpha = exp(log_negbin_alpha);
      }
      for(int j = n_layer - 1; j >= 0; j--) {

        d_h(j) = d_a(j + 1) * weight_(j + 1).t();
        if(activ == "sigmoid") {
          d_a(j) = 1/(1+exp(-a(j)));
          d_a(j) = d_h(j) % (d_a(j) % (1-d_a(j)));
        } else if(activ == "tanh") {
          d_a(j) = tanh(a(j));
          d_a(j) = d_h(j) % (1 - square(d_a(j)));
        } else if(activ == "relu") d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0));});
        else if(activ == "prelu")  d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0) + 0.2*(x <= 0));});
        else if(activ == "elu")    d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));
        else if(activ == "celu")   d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));

        if(j > 0) {
          d_w(j) = h(j - 1).t() * d_a(j);
        } else {
          d_w(j) = xi_.t() * d_a(j);
        }
        vec bias_grad = d_a(j).t() * one_sample_size;
        if(learning_rate_adaptive == "momentum") {

          last_dw(j) = last_dw(j) * rho + d_w(j)    * learning_rate;
          last_db(j) = last_db(j) * rho + bias_grad * learning_rate;
          dw(j) = last_dw(j); // d_w(j)    * learning_rate + rho * last_dw(j);
          db(j) = last_db(j); // bias_grad * learning_rate + rho * last_db(j);
        } else if (learning_rate_adaptive == "adagrad") {

          weight_ss(j) = weight_ss(j) + square(d_w(j));
          bias_ss(j)   = bias_ss(j)   + square(bias_grad);
          dw(j) = d_w(j)   /sqrt(weight_ss(j) + epsilon) * learning_rate;
          db(j) = bias_grad/sqrt(bias_ss(j)   + epsilon) * learning_rate;
        } else if (learning_rate_adaptive == "adadelta") {

          weight_egs(j) = weight_egs(j) * rho + (1-rho) * square(d_w(j));
          bias_egs(j)   = bias_egs(j)   * rho + (1-rho) * square(bias_grad);
          dw(j) = sqrt(weight_es(j) + epsilon) / sqrt(weight_egs(j) + epsilon) % d_w(j);
          db(j) = sqrt(bias_es(j)   + epsilon) / sqrt(bias_egs(j)   + epsilon) % bias_grad;
          weight_es(j) = weight_es(j) * rho + (1-rho) * square(dw(j));
          bias_es(j)   = bias_es(j)   * rho + (1-rho) * square(db(j));
        } else if (learning_rate_adaptive == "adam") {

          // adam_ind ++; // **** NEED FIX **** //
          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          dw(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(j) / (sqrt(vt_w(j) / (1-pow(beta2, adam_ind))) + epsilon);
          db(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(j) / (sqrt(vt_b(j) / (1-pow(beta2, adam_ind))) + epsilon);
        } else if (learning_rate_adaptive == "amsgrad") {

          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          cube vt_bw(vt_w(j).n_rows, vt_w(j).n_cols, 2);
          mat  vt_bb(vt_b(j).size(), 2);
          vt_bw.slice(0) = vt_w(j);
          vt_bw.slice(1) = vt_bar_w(j);
          vt_bb.col(0) = vt_b(j);
          vt_bb.col(1) = vt_bar_b(j);
          vt_bar_w(j) = max(vt_bw, 2);
          vt_bar_b(j) = max(vt_bb, 1);
          dw(j) = learning_rate * mt_w(j) / (sqrt(vt_bar_w(j)) + epsilon);
          db(j) = learning_rate * mt_b(j) / (sqrt(vt_bar_b(j)) + epsilon);
        } else {

          dw(j) = d_w(j)    * learning_rate;
          db(j) = bias_grad * learning_rate;
        }
        weight_(j) = weight_(j) - dw(j) - l1_reg * (conv_to<mat>::from(weight_(j) > 0) - conv_to<mat>::from(weight_(j) < 0)) - l2_reg * (weight_(j));
        bias_(j)   = bias_(j)   - db(j);  // column mean
      }
    }

    if(valid) {

      mat pred;
      int n_s = x_valid_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);
      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {
          pred = x_valid_ * weight_(j) + one_sample_size * bias_(j).t();
        } else {
          pred = pred * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    pred = 1/(1+exp(-pred));
        else if(activ == "tanh")  pred = tanh(pred);
        else if(activ == "relu")  pred = pred % (pred > 0);
        else if(activ == "prelu") pred = pred % (pred > 0) + (pred <= 0) % pred*0.2;
        else if(activ == "elu")   pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
        else if(activ == "celu")  pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
      }
      mat y_pred = pred * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      if(loss_f == "logit") {

        y_pred = 1 / (1 + exp(-y_pred));
        loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred) + (1-y_valid_) % log(1-y_pred), 1)) / sum(w_valid_);
      } else if(loss_f == "mse") {

        loss[k] = sum(w_valid_ % sum(pow(y_valid_ - y_pred, 2), 1)) / sum(w_valid_);
      } else if(loss_f == "cross-entropy") {

        y_pred = exp(y_pred) / (sum(exp(y_pred), 1) * one_dim_y.t());
        loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred), 1)) / sum(w_valid_);
      } else if(loss_f == "rmsle") {

        y_pred = y_pred % (y_pred > 0);
        loss[k] = sum(w_valid_ % pow(log(y_valid_ + 1) - log(y_pred + 1), 2)) / sum(w_valid_);
      } else if(loss_f == "poisson") {

        loss[k] = sum(w_valid_ % sum(-y_valid_ % y_pred + exp(y_pred), 1)) / sum(w_valid_);
      } else if(loss_f == "negbin") {

        vec alpha_inv_plus_y = y_valid_ + 1/negbin_alpha;
        loss[k] = sum(w_valid_ % sum(-alpha_inv_plus_y.transform([](double x){return(lgamma(x));}) +
          lgamma(1/negbin_alpha) + log(1 + negbin_alpha*exp(y_pred))/negbin_alpha -
          y_valid_ % (log_negbin_alpha + y_pred - log(1 + negbin_alpha*exp(y_pred))), 1)) / sum(w_valid_);
      } else if(loss_f == "poisson-nonzero") {

        loss[k] = sum(w_valid_ % sum(-y_valid_ % y_pred + exp(y_pred) +
          y_pred.transform([](double x){
            if(x < -20) {
              return(x);
            } else {
              return(log(1 - exp(-exp(x))));
            }
          }), 1)) / sum(w_valid_);
      } else if(loss_f == "negbin-nonzero") {

        vec lambda = exp(y_pred);
        vec one_plus_alpha_l = 1 + negbin_alpha*lambda;
        vec aaa = log(one_plus_alpha_l)/negbin_alpha;
        vec alpha_inv_plus_y = y_valid_ + 1/negbin_alpha;
        loss[k] = sum(w_valid_ % sum(-alpha_inv_plus_y.transform([](double x){return(lgamma(x));}) +
          lgamma(1/negbin_alpha) + aaa - y_valid_ % (log_negbin_alpha + y_pred - log(one_plus_alpha_l)) +
          log(1 - exp(-aaa)), 1)) / sum(w_valid_);
      } else if(loss_f == "zip") {

        vec y_pred_0 = 1 / (1 + exp(-y_pred.col(0)));
        vec y_pred_1 = exp(y_pred.col(1));
        loss[k] = -sum(((1 - y_valid_.col(0))%log(1 - y_pred_0%(1 - exp(-y_pred_1))) +
          y_valid_.col(0)%(log(y_pred_0) + y_valid_.col(1)%log(y_pred_1) - y_pred_1)) % w_valid_) / sum(w_valid_);
      } else if(loss_f == "zinb") {

        vec pi_ = 1 / (1 + exp(-y_pred.col(0)));
        vec lambda_ = exp(y_pred.col(1));
        vec one_plus_alpha_l = 1 + negbin_alpha*lambda_;
        vec aaa = log(one_plus_alpha_l)/negbin_alpha;
        vec y_plus_alpha_inv = y_valid_.col(1) + 1/negbin_alpha;
        loss[k] = -sum(((1 - y_valid_.col(0)) % log(1 - pi_ % (1 - exp(-aaa))) +
          y_valid_.col(0) % (log(pi_) + y_plus_alpha_inv.transform([](double x){return(lgamma(x));}) -
          lgamma(1/negbin_alpha) + y_valid_.col(1)%(log_negbin_alpha + y_pred.col(1) -
          log(one_plus_alpha_l)))) % w_valid_) / sum(w_valid_);
      }

      if(!is_finite(loss[k])) {

        break_k = k-1;
        break;
      } else {

        if(loss[k] < best_loss) {

          best_loss = loss[k];
          best_weight = weight_;
          best_bias = bias_;
          if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {
            best_negbin_alpha = negbin_alpha;
          }
        }

        if(k > early_stop_det) {
          if(prod(loss.subvec(k-early_stop_det+1, k) > loss.subvec(k-early_stop_det, k-1)) > 0) {

            break_k = k;
            break;
          }
        }
      }
    }
  }

  List best_weight_(best_weight.size());
  List best_bias_(best_weight.size());
  if(early_stop) {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(best_weight(i));
      best_bias_(i) = wrap(best_bias(i).t());
    }
  } else {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(weight_(i));
      best_bias_(i) = wrap(bias_(i).t());
    }
  }

  if((loss_f == "negbin") || (loss_f == "negbin-nonzero") || (loss_f == "zinb")) {

    List result(5);
    result(0) = best_weight_;
    result(1) = best_bias_;
    result(2) = loss;
    result(3) = break_k;
    result(4) = best_negbin_alpha;
    return(result);
  }

  List result(4);
  result(0) = best_weight_;
  result(1) = best_bias_;
  result(2) = loss;
  result(3) = break_k;
  return(result);
}
//' Backprop Long (Internal)
//'
//' @param n_hidden Hidden layer numbers.
//' @param w_ini Initial weights.
//' @param weight_pathway Pathway weights.
//' @param bias_pathway Pathway Biases.
//' @param l1_pathway L1-penalty for pathway.
//' @param l2_pathway L2-penalty for pathway.
//' @param load_param Whether to load parameters.
//' @param weight Weights.
//' @param bias Biases.
//' @param x Training.
//' @param y Training.
//' @param w Training.
//' @param x_pathway Training.
//' @param valid Whether to use a validation set.
//' @param x_valid Validation.
//' @param y_valid Validation.
//' @param w_valid Validation.
//' @param x_valid_pathway Validation.
//' @param activ Activation function.
//' @param activ_pathway Activation function for pathway.
//' @param n_epoch Number of epochs.
//' @param n_batch Batch size.
//' @param model_type Model type.
//' @param learning_rate (Initial) learning rate.
//' @param l1_reg L1-penalty.
//' @param l2_reg L2-penalty.
//' @param early_stop Whether to early stop.
//' @param early_stop_det Number of epochs to determine early-stop.
//' @param learning_rate_adaptive Adaptive learning rate adjustment method.
//' @param rho Parameter.
//' @param epsilon Parameter.
//' @param beta1 Parameter.
//' @param beta2 Parameter.
//' @param loss_f Loss function.
//'
//' @return A list of outputs
//'
// [[Rcpp::export]]
SEXP backprop_long(NumericVector n_hidden, double w_ini,
                   List weight_pathway, NumericVector bias_pathway,
                   double l1_pathway, double l2_pathway,
                   bool load_param, List weight, List bias,
                   NumericMatrix x, NumericMatrix y, NumericVector w, List x_pathway, bool valid,
                   NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid, List x_valid_pathway,
                   std::string activ, std::string activ_pathway,
                   int n_epoch, int n_batch, std::string model_type,
                   double learning_rate, double l1_reg, double l2_reg, bool early_stop, int early_stop_det,
                   std::string learning_rate_adaptive, double rho, double epsilon, double beta1, double beta2,
                   std::string loss_f) {

  //Rcout << "Start" << endl;
  mat x_ = as<mat>(x);
  mat y_ = as<mat>(y);
  vec w_ = as<vec>(w);
  unsigned int sample_size = x.nrow();
  int n_layer = n_hidden.size();
  int n_pathway = x_pathway.size();
  field<mat> x_pathway_(n_pathway);
  field<vec> d_w_pathway(n_pathway);
  field<vec> weight_pathway_(n_pathway);
  vec bias_pathway_(n_pathway);

  field<mat> weight_(n_layer + 1);
  field<vec> bias_(n_layer + 1);
  field<mat> a(n_layer + 1);
  field<mat> h(n_layer + 1);
  field<mat> d_a(n_layer + 1);
  field<mat> d_h(n_layer + 1);
  field<mat> d_w(n_layer + 1);
  vec loss(n_epoch);
  double best_loss = INFINITY;
  field<mat> best_weight(n_layer + 1);
  field<vec> best_bias(n_layer + 1);
  field<vec> best_weight_pathway(n_pathway);
  vec best_bias_pathway(n_pathway);
  int break_k = n_epoch - 1;

  // adapative learning rate
  field<mat> dw(n_layer + 1);
  field<vec> db(n_layer + 1);
  field<vec> dw_pathway(n_pathway);
  vec db_pathway(n_pathway);
  // momentum
  field<mat> last_dw(n_layer + 1);
  field<vec> last_db(n_layer + 1);
  field<vec> last_dw_pathway(n_pathway);
  vec last_db_pathway(n_pathway);
  // adagrad
  field<mat> weight_ss(n_layer + 1);
  field<vec> bias_ss(n_layer + 1);
  field<vec> weight_ss_pathway(n_pathway);
  vec bias_ss_pathway(n_pathway);
  // adadelta
  field<mat> weight_egs(n_layer + 1);
  field<vec> bias_egs(n_layer + 1);
  field<mat> weight_es(n_layer + 1);
  field<vec> bias_es(n_layer + 1);
  field<vec> weight_egs_pathway(n_pathway);
  vec bias_egs_pathway(n_pathway);
  field<vec> weight_es_pathway(n_pathway);
  vec bias_es_pathway(n_pathway);
  // adam
  field<mat> mt_w(n_layer + 1);
  field<vec> mt_b(n_layer + 1);
  field<mat> vt_w(n_layer + 1);
  field<vec> vt_b(n_layer + 1);
  field<vec> mt_w_pathway(n_pathway);
  vec mt_b_pathway(n_pathway);
  field<vec> vt_w_pathway(n_pathway);
  vec vt_b_pathway(n_pathway);
  // amsgrad
  field<mat> vt_bar_w(n_layer + 1);
  field<vec> vt_bar_b(n_layer + 1);
  field<vec> vt_bar_w_pathway(n_pathway);
  vec vt_bar_b_pathway(n_pathway);
  double adam_ind = 0;

  //Rcout << "Initiate" << endl;
  for(int i = 0; i < n_pathway; i++) {
    x_pathway_(i) = as<mat>(x_pathway(i));
    if(load_param) {
      weight_pathway_(i) = as<vec>(weight_pathway(i));
      bias_pathway_[i] = bias_pathway[i];
    } else {
      weight_pathway_(i) = (randu<vec>(x_pathway_(i).n_cols) - 0.5) * w_ini;
      bias_pathway_[i] = (randu<double>() - 0.5) * w_ini;
    }
    dw_pathway(i) = vec(n_pathway, fill::zeros);
    db_pathway[i] = 0;
  }

  mat x_valid_;
  mat y_valid_;
  vec w_valid_;
  field<mat> x_valid_pathway_(n_pathway);
  if(valid) {

    x_valid_ = as<mat>(x_valid);
    y_valid_ = as<mat>(y_valid);
    w_valid_ = as<vec>(w_valid);
    for(int i = 0; i < n_pathway; i++) {
      x_valid_pathway_(i) = as<mat>(x_valid_pathway(i));
    }
  }

  for(int i = 0; i < n_layer + 1; i++) {

    if(i == 0) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(x_.n_cols, n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    } else if(i == n_layer) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], y_.n_cols) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(y_.n_cols) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
      db(i) = vec(y_.n_cols, fill::zeros);
    } else {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    }
  }

  best_weight = weight_;
  best_bias = bias_;

  if(learning_rate_adaptive == "momentum") {

    // MOMEMTUM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        last_dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        last_dw(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        last_db(i) = vec(y_.n_cols, fill::zeros);
      } else {
        last_dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      }
    }
    for(int i = 0; i < n_pathway; i++) {
      last_dw_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      last_db_pathway[i] = 0;
    }
  } else if(learning_rate_adaptive == "adagrad") {

    // ADAGRAD
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_ss(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_ss(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_ss(i) = vec(y_.n_cols, fill::zeros);
      } else {
        weight_ss(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      }
    }
    for(int i = 0; i < n_pathway; i++) {
      weight_ss_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      bias_ss_pathway[i] = 0;
    }
  } else if(learning_rate_adaptive == "adadelta") {

    // ADADELTA
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_egs(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_egs(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_egs(i) = vec(y_.n_cols, fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        bias_es(i) = vec(y_.n_cols, fill::zeros);
      } else {
        weight_egs(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      }
    }
    for(int i = 0; i < n_pathway; i++) {
      weight_egs_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      bias_egs_pathway[i] = 0;
      weight_es_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      bias_es_pathway[i] = 0;
    }
  } else if(learning_rate_adaptive == "adam") {

    // ADAM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        mt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_b(i) = vec(y_.n_cols, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
    for(int i = 0; i < n_pathway; i++) {
      mt_w_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      mt_b_pathway[i] = 0;
      vt_w_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      vt_b_pathway[i] = 0;
    }
  } else if(learning_rate_adaptive == "amsgrad") {

    // AMSGrad
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        mt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_b(i) = vec(y_.n_cols, fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
        vt_bar_b(i) = vec(y_.n_cols, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
    for(int i = 0; i < n_pathway; i++) {
      mt_w_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      mt_b_pathway[i] = 0;
      vt_w_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      vt_b_pathway[i] = 0;
      vt_bar_w_pathway(i) = vec(x_pathway_(i).n_cols, fill::zeros);
      vt_bar_b_pathway[i] = 0;
    }
  }

  int n_round = ceil((sample_size - 1)/n_batch);
  uvec i_bgn(n_round);
  uvec i_end(n_round);

  for(int s = 0; s < n_round; s++) {

    i_bgn[s] = s*n_batch;
    i_end[s] = (s+1)*n_batch - 1;
    if(i_end[s] > sample_size - 1) i_end[s] = sample_size - 1;
    if(s == n_round - 1) i_end[s] = sample_size - 1;
  }

  //Rcout << "Ready" << endl;
  for(int k = 0; k < n_epoch; k++) {

    //Rcout << k << endl;
    // shuffle
    uvec new_order = as<uvec>(sample(sample_size, sample_size)) - 1;
    x_ = x_.rows(new_order);
    y_ = y_.rows(new_order);
    w_ = w_.elem(new_order);
    for(int j = 0; j < n_pathway; j++)
      x_pathway_(j) = x_pathway_(j).rows(new_order);

    for(int i = 0; i < n_round; i++) {

      //Rcout << i << ' ';
      mat xi_ = x_.rows(i_bgn[i], i_end[i]);
      mat yi_ = y_.rows(i_bgn[i], i_end[i]);
      vec wi_ = w_.subvec(i_bgn[i], i_end[i]);
      int n_s = xi_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);
      field<mat> xi_pathway_(n_pathway);

      mat a_pathway(n_s, n_pathway);
      mat d_h_pathway(n_s, n_pathway);
      mat d_a_pathway(n_s, n_pathway);

      for(int j = 0; j < n_pathway; j++) {
        //Rcout << j << ' ';
        xi_pathway_(j) = x_pathway_(j).rows(i_bgn[i], i_end[i]);
        a_pathway.col(j) = xi_pathway_(j) * weight_pathway_(j) + bias_pathway_[j];
        if(activ_pathway == "sigmoid")    xi_.col(j) = 1/(1+exp(-a_pathway.col(j)));
        else if(activ_pathway == "tanh")  xi_.col(j) = tanh(a_pathway.col(j));
        else if(activ_pathway == "relu")  xi_.col(j) = a_pathway.col(j) % (a_pathway.col(j) > 0);
        else if(activ_pathway == "prelu") xi_.col(j) = a_pathway.col(j) % (a_pathway.col(j) > 0) + (a_pathway.col(j) <= 0) % a_pathway.col(j)*0.2;
        else if(activ_pathway == "elu")   xi_.col(j) = a_pathway.col(j) % (a_pathway.col(j) > 0) + exp((a_pathway.col(j) <= 0) % a_pathway.col(j)) - 1;
        else if(activ_pathway == "celu")  xi_.col(j) = a_pathway.col(j) % (a_pathway.col(j) > 0) + exp((a_pathway.col(j) <= 0) % a_pathway.col(j)) - 1;
        else xi_.col(j) = a_pathway.col(j);
      }
      for(int j = 0; j < n_layer; j++) {

        //Rcout << j << ' ';
        if(j == 0) {

          a(j) = xi_ * weight_(j) + one_sample_size * bias_(j).t();
        } else{

          a(j) = h(j-1) * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    h(j) = 1/(1+exp(-a(j)));
        else if(activ == "tanh")  h(j) = tanh(a(j));
        else if(activ == "relu")  h(j) = a(j) % (a(j) > 0);
        else if(activ == "prelu") h(j) = a(j) % (a(j) > 0) + (a(j) <= 0) % a(j)*0.2;
        else if(activ == "elu")   h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
        else if(activ == "celu")  h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
      }
      mat y_pi = h(n_layer - 1) * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      if(loss_f == "logit")
        y_pi = 1 / (1 + exp(-y_pi));

      if(loss_f == "rmsle") {

        y_pi = y_pi % (y_pi > 0);
        d_a(n_layer) = -(log(yi_ + 1) - log(y_pi + 1)) /
          (y_pi + 1) % y_pi.transform([](double x){return(1*(x > 0));}) % wi_ / sum(wi_);
      } else if(loss_f == "cross-entropy") {

        y_pi = exp(y_pi) / (sum(exp(y_pi), 1) * one_dim_y.t());
        d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      } else {

        // d_a(n_layer) = -(yi_ - y_pi) % wi_ / sum(wi_);
        d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      }

      d_w(n_layer) = h(n_layer - 1).t() * d_a(n_layer);
      vec bias_grad = d_a(n_layer).t() * one_sample_size;
      if(learning_rate_adaptive == "momentum") {

        last_dw(n_layer) = last_dw(n_layer) * rho + d_w(n_layer) * learning_rate;
        last_db(n_layer) = last_db(n_layer) * rho + bias_grad    * learning_rate;
        dw(n_layer) = last_dw(n_layer);
        db(n_layer) = last_db(n_layer);
      } else if (learning_rate_adaptive == "adagrad") {

        weight_ss(n_layer) = weight_ss(n_layer) + square(d_w(n_layer));
        bias_ss(n_layer)   = bias_ss(n_layer)   + square(bias_grad);
        dw(n_layer) = d_w(n_layer)/sqrt(weight_ss(n_layer) + epsilon) * learning_rate;
        db(n_layer) = bias_grad   /sqrt(bias_ss(n_layer)   + epsilon) * learning_rate;
      } else if (learning_rate_adaptive == "adadelta") {

        weight_egs(n_layer) = weight_egs(n_layer) * rho + (1-rho) * square(d_w(n_layer));
        bias_egs(n_layer)   = bias_egs(n_layer)   * rho + (1-rho) * square(bias_grad);
        dw(n_layer) = sqrt(weight_es(n_layer) + epsilon) / sqrt(weight_egs(n_layer) + epsilon) % d_w(n_layer);
        db(n_layer) = sqrt(bias_es(n_layer)   + epsilon) / sqrt(bias_egs(n_layer)   + epsilon) % bias_grad;
        weight_es(n_layer) = weight_es(n_layer) * rho + (1-rho) * square(dw(n_layer));
        bias_es(n_layer)   = bias_es(n_layer)   * rho + (1-rho) * square(db(n_layer));
      } else if (learning_rate_adaptive == "adam") {

        adam_ind = adam_ind + 1;
        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        dw(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(n_layer) / (sqrt(vt_w(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
        db(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(n_layer) / (sqrt(vt_b(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
      } else if (learning_rate_adaptive == "amsgrad") {

        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        cube vt_bw(vt_w(n_layer).n_rows, vt_w(n_layer).n_cols, 2);
        mat  vt_bb(vt_b(n_layer).size(), 2);
        vt_bw.slice(0) = vt_w(n_layer);
        vt_bw.slice(1) = vt_bar_w(n_layer);
        vt_bb.col(0) = vt_b(n_layer);
        vt_bb.col(1) = vt_bar_b(n_layer);
        vt_bar_w(n_layer) = max(vt_bw, 2);
        vt_bar_b(n_layer) = max(vt_bb, 1);
        dw(n_layer) = learning_rate * mt_w(n_layer) / (sqrt(vt_bar_w(n_layer)) + epsilon);
        db(n_layer) = learning_rate * mt_b(n_layer) / (sqrt(vt_bar_b(n_layer)) + epsilon);
      } else {

        dw(n_layer) = d_w(n_layer) * learning_rate;
        db(n_layer) = bias_grad    * learning_rate;
      }
      weight_(n_layer) = weight_(n_layer) - dw(n_layer) - l1_reg * (conv_to<mat>::from(weight_(n_layer) > 0) - conv_to<mat>::from(weight_(n_layer) < 0)) - l2_reg * (weight_(n_layer));
      bias_(n_layer)   = bias_(n_layer)   - db(n_layer);
      for(int j = n_layer - 1; j >= 0; j--) {

        d_h(j) = d_a(j + 1) * weight_(j + 1).t();
        if(activ == "sigmoid") {
          d_a(j) = 1/(1+exp(-a(j)));
          d_a(j) = d_h(j) % (d_a(j) % (1-d_a(j)));
        } else if(activ == "tanh") {
          d_a(j) = tanh(a(j));
          d_a(j) = d_h(j) % (1 - square(d_a(j)));
        } else if(activ == "relu") d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0));});
        else if(activ == "prelu")  d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0) + 0.2*(x <= 0));});
        else if(activ == "elu")    d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));
        else if(activ == "celu")   d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));

        if(j > 0) {
          d_w(j) = h(j - 1).t() * d_a(j);
        } else {
          d_w(j) = xi_.t() * d_a(j);
        }
        vec bias_grad = d_a(j).t() * one_sample_size;
        if(learning_rate_adaptive == "momentum") {

          last_dw(j) = last_dw(j) * rho + d_w(j)    * learning_rate;
          last_db(j) = last_db(j) * rho + bias_grad * learning_rate;
          dw(j) = d_w(j)    * learning_rate + rho * last_dw(j);
          db(j) = bias_grad * learning_rate + rho * last_db(j);
        } else if (learning_rate_adaptive == "adagrad") {

          weight_ss(j) = weight_ss(j) + square(d_w(j));
          bias_ss(j)   = bias_ss(j)   + square(bias_grad);
          dw(j) = d_w(j)   /sqrt(weight_ss(j) + epsilon) * learning_rate;
          db(j) = bias_grad/sqrt(bias_ss(j)   + epsilon) * learning_rate;
        } else if (learning_rate_adaptive == "adadelta") {

          weight_egs(j) = weight_egs(j) * rho + (1-rho) * square(d_w(j));
          bias_egs(j)   = bias_egs(j)   * rho + (1-rho) * square(bias_grad);
          dw(j) = sqrt(weight_es(j) + epsilon) / sqrt(weight_egs(j) + epsilon) % d_w(j);
          db(j) = sqrt(bias_es(j)   + epsilon) / sqrt(bias_egs(j)   + epsilon) % bias_grad;
          weight_es(j) = weight_es(j) * rho + (1-rho) * square(dw(j));
          bias_es(j)   = bias_es(j)   * rho + (1-rho) * square(db(j));
        } else if (learning_rate_adaptive == "adam") {

          // adam_ind ++;
          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          dw(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(j) / (sqrt(vt_w(j) / (1-pow(beta2, adam_ind))) + epsilon);
          db(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(j) / (sqrt(vt_b(j) / (1-pow(beta2, adam_ind))) + epsilon);
        } else if (learning_rate_adaptive == "amsgrad") {

          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          cube vt_bw(vt_w(j).n_rows, vt_w(j).n_cols, 2);
          mat  vt_bb(vt_b(j).size(), 2);
          vt_bw.slice(0) = vt_w(j);
          vt_bw.slice(1) = vt_bar_w(j);
          vt_bb.col(0) = vt_b(j);
          vt_bb.col(1) = vt_bar_b(j);
          vt_bar_w(j) = max(vt_bw, 2);
          vt_bar_b(j) = max(vt_bb, 1);
          dw(j) = learning_rate * mt_w(j) / (sqrt(vt_bar_w(j)) + epsilon);
          db(j) = learning_rate * mt_b(j) / (sqrt(vt_bar_b(j)) + epsilon);
        } else {

          dw(j) = d_w(j)    * learning_rate;
          db(j) = bias_grad * learning_rate;
        }
        weight_(j) = weight_(j) - dw(j) - l1_reg * (conv_to<mat>::from(weight_(j) > 0) - conv_to<mat>::from(weight_(j) < 0)) - l2_reg * (weight_(j));
        bias_(j)   = bias_(j)   - db(j);  // column mean
      }

      //Rcout << "z" << ' ';
      d_h_pathway = d_a(0) * weight_(0).t();
      //Rcout << "y" << ' ';
      for(int j = 0; j < n_pathway; j++) {

        //Rcout << j << ' ';
        if(activ_pathway == "sigmoid") {
          d_a_pathway.col(j) = 1/(1+exp(-a_pathway.col(j)));
          d_a_pathway.col(j) = d_h_pathway.col(j) % (d_a_pathway.col(j) % (1-d_a_pathway.col(j)));
        } else if(activ_pathway == "tanh") {
          d_a_pathway.col(j) = tanh(a_pathway.col(j));
          d_a_pathway.col(j) = d_h_pathway.col(j) % (1 - square(d_a_pathway.col(j)));
        } else if(activ_pathway == "relu")
          d_a_pathway.col(j) = d_h_pathway.col(j) % a_pathway.transform([](double x){return(1*(x > 0));}).col(j);
        else if(activ_pathway == "prelu")
          d_a_pathway.col(j) = d_h_pathway.col(j) % a_pathway.transform([](double x){return(1*(x > 0) + 0.2*(x <= 0));}).col(j);
        else if(activ_pathway == "elu")
          d_a_pathway.col(j) = d_h_pathway.col(j) % ((a_pathway.col(j) > 0) +
                (a_pathway.col(j) <= 0) % exp((a_pathway.col(j) <= 0) % a_pathway.col(j)));
        else if(activ_pathway == "celu")
          d_a_pathway.col(j) = d_h_pathway.col(j) % ((a_pathway.col(j) > 0) +
                (a_pathway.col(j) <= 0) % exp((a_pathway.col(j) <= 0) % a_pathway.col(j)));
        else
          d_a_pathway.col(j) = d_h_pathway.col(j);

        //Rcout << "a" << ' ';
        d_w_pathway(j) = xi_pathway_(j).t() * d_a_pathway.col(j);
        //Rcout << "b" << ' ';
        double bias_grad = sum(d_a_pathway.col(j) % one_sample_size);
        //Rcout << "c" << ' ';
        if(learning_rate_adaptive == "momentum") {

          last_dw_pathway(j) = last_dw_pathway(j) * rho + d_w_pathway(j)    * learning_rate;
          last_db_pathway[j] = last_db_pathway[j] * rho + bias_grad * learning_rate;
          dw_pathway(j) = d_w_pathway(j)    * learning_rate + rho * last_dw_pathway(j);
          db_pathway[j] = bias_grad * learning_rate + rho * last_db_pathway[j];
        } else if (learning_rate_adaptive == "adagrad") {

          weight_ss_pathway(j) = weight_ss_pathway(j) + square(d_w_pathway(j));
          bias_ss_pathway[j]   = bias_ss_pathway[j]   + bias_grad * bias_grad;
          dw_pathway(j) = d_w_pathway(j)   /sqrt(weight_ss_pathway(j) + epsilon) * learning_rate;
          db_pathway[j] = bias_grad/sqrt(bias_ss_pathway[j]   + epsilon) * learning_rate;
        } else if (learning_rate_adaptive == "adadelta") {

          weight_egs_pathway(j) = weight_egs_pathway(j) * rho + (1-rho) * square(d_w_pathway(j));
          bias_egs_pathway[j]   = bias_egs_pathway[j]   * rho + (1-rho) * bias_grad * bias_grad;
          dw_pathway(j) = sqrt(weight_es_pathway(j) + epsilon) / sqrt(weight_egs_pathway(j) + epsilon) % d_w_pathway(j);
          db_pathway[j] = sqrt(bias_es_pathway[j]   + epsilon) / sqrt(bias_egs_pathway[j]   + epsilon) * bias_grad;
          weight_es_pathway(j) = weight_es_pathway(j) * rho + (1-rho) * square(dw_pathway(j));
          bias_es_pathway[j]   = bias_es_pathway[j]   * rho + (1-rho) * db_pathway[j] * db_pathway[j];
        } else if (learning_rate_adaptive == "adam") {

          // adam_ind ++;
          mt_w_pathway(j) = mt_w_pathway(j) * beta1 + (1-beta1) * d_w_pathway(j);
          mt_b_pathway[j] = mt_b_pathway[j] * beta1 + (1-beta1) * bias_grad;
          vt_w_pathway(j) = vt_w_pathway(j) * beta2 + (1-beta2) * square(d_w_pathway(j));
          vt_b_pathway[j] = vt_b_pathway[j] * beta2 + (1-beta2) * bias_grad * bias_grad;
          dw_pathway(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w_pathway(j) / (sqrt(vt_w_pathway(j) / (1-pow(beta2, adam_ind))) + epsilon);
          db_pathway[j] = learning_rate / (1-pow(beta1, adam_ind)) * mt_b_pathway[j] / (sqrt(vt_b_pathway[j] / (1-pow(beta2, adam_ind))) + epsilon);
        } else if (learning_rate_adaptive == "amsgrad") {

          mt_w_pathway(j) = mt_w_pathway(j) * beta1 + (1-beta1) * d_w_pathway(j);
          mt_b_pathway[j] = mt_b_pathway[j] * beta1 + (1-beta1) * bias_grad;
          vt_w_pathway(j) = vt_w_pathway(j) * beta2 + (1-beta2) * square(d_w_pathway(j));
          vt_b_pathway[j] = vt_b_pathway[j] * beta2 + (1-beta2) * bias_grad * bias_grad;
          mat vt_bw_pathway(vt_w_pathway(j).size(), 2);
          vec vt_bb_pathway(2);
          vt_bw_pathway.col(0) = vt_w_pathway(j);
          vt_bw_pathway.col(1) = vt_bar_w_pathway(j);
          vt_bb_pathway[0] = vt_b_pathway[j];
          vt_bb_pathway[1] = vt_bar_b_pathway[j];
          vt_bar_w_pathway(j) = max(vt_bw_pathway, 1);
          vt_bar_b_pathway[j] = max(vt_bb_pathway);
          dw_pathway(j) = learning_rate * mt_w_pathway(j) / (sqrt(vt_bar_w_pathway(j)) + epsilon);
          db_pathway[j] = learning_rate * mt_b_pathway[j] / (sqrt(vt_bar_b_pathway[j]) + epsilon);
        } else {

          dw_pathway(j) = d_w_pathway(j)    * learning_rate;
          db_pathway[j] = bias_grad * learning_rate;
        }
        //Rcout << "d" << ' ';
        weight_pathway_(j) = weight_pathway_(j) - dw_pathway(j) -
          l1_pathway * (conv_to<vec>::from(weight_pathway_(j) > 0) - conv_to<vec>::from(weight_pathway_(j) < 0)) -
          l2_pathway * (weight_pathway_(j));
        bias_pathway_[j]   = bias_pathway_[j]   - db_pathway[j];
      }
    }

    if(valid) {

      mat pred;
      int n_s = x_valid_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);
      for(int j = 0; j < n_layer; j++) {

        for(int i = 0; i < n_pathway; i++) {
          x_valid_.col(i) = x_valid_pathway_(i) * weight_pathway_(i) + bias_pathway_[i];
          if(activ_pathway == "sigmoid")    x_valid_.col(i) = 1/(1+exp(-x_valid_.col(i)));
          else if(activ_pathway == "tanh")  x_valid_.col(i) = tanh(x_valid_.col(i));
          else if(activ_pathway == "relu")  x_valid_.col(i) = x_valid_.col(i) % (x_valid_.col(i) > 0);
          else if(activ_pathway == "prelu") x_valid_.col(i) = x_valid_.col(i) % (x_valid_.col(i) > 0) + (x_valid_.col(i) <= 0) % x_valid_.col(i)*0.2;
          else if(activ_pathway == "elu")   x_valid_.col(i) = x_valid_.col(i) % (x_valid_.col(i) > 0) + exp((x_valid_.col(i) <= 0) % x_valid_.col(i)) - 1;
          else if(activ_pathway == "celu")  x_valid_.col(i) = x_valid_.col(i) % (x_valid_.col(i) > 0) + exp((x_valid_.col(i) <= 0) % x_valid_.col(i)) - 1;
        }
        if(j == 0) {
          pred = x_valid_ * weight_(j) + one_sample_size * bias_(j).t();
        } else {
          pred = pred * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    pred = 1/(1+exp(-pred));
        else if(activ == "tanh")  pred = tanh(pred);
        else if(activ == "relu")  pred = pred % (pred > 0);
        else if(activ == "prelu") pred = pred % (pred > 0) + (pred <= 0) % pred*0.2;
        else if(activ == "elu")   pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
        else if(activ == "celu")  pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
      }
      mat y_pred = pred * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      if(loss_f == "logit") {

        y_pred = 1 / (1 + exp(-y_pred));
        loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred) + (1-y_valid_) % log(1-y_pred), 1)) / sum(w_valid_);
      } else if(loss_f == "mse") {

        loss[k] = sum(w_valid_ % sum(pow(y_valid_ - y_pred, 2), 1)) / sum(w_valid_);
      } else if(loss_f == "cross-entropy") {

        y_pred = exp(y_pred) / (sum(exp(y_pred), 1) * one_dim_y.t());
        loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred), 1)) / sum(w_valid_);
      } else if(loss_f == "rmsle") {

        y_pred = y_pred % (y_pred > 0);
        loss[k] = sum(w_valid_ % pow(log(y_valid_ + 1) - log(y_pred + 1), 2)) / sum(w_valid_);
      }

      if(!is_finite(loss[k])) {

        break_k = k-1;
        break;
      } else {

        if(loss[k] < best_loss) {

          best_loss = loss[k];
          best_weight = weight_;
          best_bias = bias_;
          best_weight_pathway = weight_pathway_;
          best_bias_pathway = bias_pathway_;
        }

        if(k > early_stop_det) {
          if(prod(loss.subvec(k-early_stop_det+1, k) > loss.subvec(k-early_stop_det, k-1)) > 0) {

            break_k = k;
            break;
          }
        }
      }
    }
  }

  List best_weight_(best_weight.size());
  List best_bias_(best_weight.size());
  List best_weight_pathway_(n_pathway);
  NumericVector best_bias_pathway_(n_pathway);
  if(early_stop) {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(best_weight(i));
      best_bias_(i) = wrap(best_bias(i).t());
    }
    for(int i = 0; i < n_pathway; i++) {
      best_weight_pathway_(i) = wrap(best_weight_pathway(i));
      best_bias_pathway_[i] = best_bias_pathway[i];
    }
  } else {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(weight_(i));
      best_bias_(i) = wrap(bias_(i).t());
    }
    for(int i = 0; i < n_pathway; i++) {
      best_weight_pathway_(i) = wrap(weight_pathway(i));
      best_bias_pathway_[i] = bias_pathway[i];
    }
  }

  List result(6);
  result(0) = best_weight_;
  result(1) = best_bias_;
  result(2) = loss;
  result(3) = break_k;
  result(4) = best_weight_pathway_;
  result(5) = best_bias_pathway_;
  return(result);
}
//' Backprop Surv (Internal)
//'
//' @param n_hidden Hidden layer numbers.
//' @param w_ini Initial weights.
//' @param load_param Whether to load parameters.
//' @param weight Weights.
//' @param bias Biases.
//' @param x Training.
//' @param y Training.
//' @param w Training.
//' @param valid Whether to use a validation set.
//' @param x_valid Validation.
//' @param y_valid Validation.
//' @param w_valid Validation.
//' @param activ Activation function.
//' @param n_epoch Number of epochs.
//' @param n_batch Batch size.
//' @param model_type Model type.
//' @param learning_rate (Initial) learning rate.
//' @param l1_reg L1-penalty.
//' @param l2_reg L2-penalty.
//' @param early_stop Whether to early stop.
//' @param early_stop_det Number of epochs to determine early-stop.
//' @param learning_rate_adaptive Adaptive learning rate adjustment method.
//' @param rho Parameter.
//' @param epsilon Parameter.
//' @param beta1 Parameter.
//' @param beta2 Parameter.
//' @param loss_f Loss function.
//'
//' @return A list of outputs
//'
// [[Rcpp::export]]
SEXP backprop_surv(NumericVector n_hidden, double w_ini, // List weight, List bias,
                   bool load_param, List weight, List bias,
                   NumericMatrix x, NumericMatrix y, NumericVector w, bool valid,
                   NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid,
                   std::string activ,
                   int n_epoch, int n_batch, std::string model_type,
                   double learning_rate, double l1_reg, double l2_reg, bool early_stop, int early_stop_det,
                   std::string learning_rate_adaptive, double rho, double epsilon, double beta1, double beta2,
                   std::string loss_f) {

  mat x_ = as<mat>(x);
  mat y_ = as<mat>(y);
  vec w_ = as<vec>(w);
  mat x_valid_;
  mat y_valid_;
  vec w_valid_;

  if(valid) {

    x_valid_ = as<mat>(x_valid);
    y_valid_ = as<mat>(y_valid);
    w_valid_ = as<vec>(w_valid);

    uvec valid_order_ = sort_index(y_valid_.col(0));
    x_valid_ = x_valid_.rows(valid_order_);
    y_valid_ = y_valid_.rows(valid_order_);
    w_valid_ = w_valid_.elem(valid_order_);
  }

  unsigned int sample_size = x.nrow();
  int n_layer = n_hidden.size();
  field<mat> weight_(n_layer + 1);
  field<vec> bias_(n_layer + 1);
  field<mat> a(n_layer + 1);
  field<mat> h(n_layer + 1);
  field<mat> d_a(n_layer + 1);
  field<mat> d_h(n_layer + 1);
  field<mat> d_w(n_layer + 1);
  vec loss(n_epoch);
  double best_loss = INFINITY;
  field<mat> best_weight(n_layer + 1);
  field<vec> best_bias(n_layer + 1);
  int break_k = n_epoch - 1;

  // adapative learning rate
  field<mat> dw(n_layer + 1);
  field<vec> db(n_layer + 1);
  // momentum
  field<mat> last_dw(n_layer + 1);
  field<vec> last_db(n_layer + 1);
  // adagrad
  field<mat> weight_ss(n_layer + 1);
  field<vec> bias_ss(n_layer + 1);
  // adadelta
  field<mat> weight_egs(n_layer + 1);
  field<vec> bias_egs(n_layer + 1);
  field<mat> weight_es(n_layer + 1);
  field<vec> bias_es(n_layer + 1);
  // adam
  field<mat> mt_w(n_layer + 1);
  field<vec> mt_b(n_layer + 1);
  field<mat> vt_w(n_layer + 1);
  field<vec> vt_b(n_layer + 1);
  // amsgrad
  field<mat> vt_bar_w(n_layer + 1);
  field<vec> vt_bar_b(n_layer + 1);
  double adam_ind = 0;

  for(int i = 0; i < n_layer + 1; i++) {

    if(i == 0) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(x_.n_cols, n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    } else if(i == n_layer) {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], 1) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(1) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], y_.n_cols, fill::zeros);
      db(i) = vec(y_.n_cols, fill::zeros);
    } else {
      if(load_param) {

        weight_(i) = as<mat>(weight(i));
        bias_(i) = as<vec>(bias(i));
      } else {

        weight_(i) = (randu<mat>(n_hidden[i-1], n_hidden[i]) - 0.5) * w_ini * 2;
        bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
      }
      dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
      db(i) = vec(n_hidden[i], fill::zeros);
    }
  }

  best_weight = weight_;
  best_bias = bias_;

  if(learning_rate_adaptive == "momentum") {

    // MOMEMTUM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        last_dw(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        last_dw(i) = mat(n_hidden[i-1], 1, fill::zeros);
        last_db(i) = vec(1, fill::zeros);
      } else {
        last_dw(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        last_db(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adagrad") {

    // ADAGRAD
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_ss(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_ss(i) = mat(n_hidden[i-1], 1, fill::zeros);
        bias_ss(i) = vec(1, fill::zeros);
      } else {
        weight_ss(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_ss(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adadelta") {

    // ADADELTA
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        weight_egs(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        weight_egs(i) = mat(n_hidden[i-1], 1, fill::zeros);
        bias_egs(i) = vec(1, fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], 1, fill::zeros);
        bias_es(i) = vec(1, fill::zeros);
      } else {
        weight_egs(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_egs(i) = vec(n_hidden[i], fill::zeros);
        weight_es(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        bias_es(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "adam") {

    // ADAM
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], 1, fill::zeros);
        mt_b(i) = vec(1, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], 1, fill::zeros);
        vt_b(i) = vec(1, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  } else if(learning_rate_adaptive == "amsgrad") {

    // AMSGrad
    for(int i = 0; i < n_layer + 1; i++) {

      if(i == 0) {
        mt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(x_.n_cols, n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      } else if(i == n_layer) {
        mt_w(i) = mat(n_hidden[i-1], 1, fill::zeros);
        mt_b(i) = vec(1, fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], 1, fill::zeros);
        vt_b(i) = vec(1, fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], 1, fill::zeros);
        vt_bar_b(i) = vec(1, fill::zeros);
      } else {
        mt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        mt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_b(i) = vec(n_hidden[i], fill::zeros);
        vt_bar_w(i) = mat(n_hidden[i-1], n_hidden[i], fill::zeros);
        vt_bar_b(i) = vec(n_hidden[i], fill::zeros);
      }
    }
  }

  int n_round = ceil((sample_size - 1)/n_batch);
  uvec i_bgn(n_round);
  uvec i_end(n_round);

  for(int s = 0; s < n_round; s++) {

    i_bgn[s] = s*n_batch;
    i_end[s] = (s+1)*n_batch - 1;
    if(i_end[s] > sample_size - 1) i_end[s] = sample_size - 1;
    if(s == n_round - 1) i_end[s] = sample_size - 1;
  }

  for(int k = 0; k < n_epoch; k++) {

    // shuffle
    uvec new_order = as<uvec>(sample(sample_size, sample_size)) - 1;
    x_ = x_.rows(new_order);
    y_ = y_.rows(new_order);
    w_ = w_.elem(new_order);

    for(int i = 0; i < n_round; i++) {

      mat xi_ = x_.rows(i_bgn[i], i_end[i]);
      mat yi_ = y_.rows(i_bgn[i], i_end[i]);
      vec wi_ = w_.subvec(i_bgn[i], i_end[i]);

      uvec ind_yi_ = sort_index(yi_.col(0));
      xi_ = xi_.rows(ind_yi_);
      yi_ = yi_.rows(ind_yi_);
      wi_ = wi_.elem(ind_yi_);

      int n_s = xi_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);

      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {

          a(j) = xi_ * weight_(j) + one_sample_size * bias_(j).t();
        } else{

          a(j) = h(j-1) * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    h(j) = 1/(1+exp(-a(j)));
        else if(activ == "tanh")  h(j) = tanh(a(j));
        else if(activ == "relu")  h(j) = a(j) % (a(j) > 0);
        else if(activ == "prelu") h(j) = a(j) % (a(j) > 0) + (a(j) <= 0) % a(j)*0.2;
        else if(activ == "elu")   h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
        else if(activ == "celu")  h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
      }
      mat y_pi = h(n_layer - 1) * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      mat temp(n_s, 1);
      double curr = sum(sum(exp(y_pi)));
      double curr2 = 0;
      for(int j = 0; j < n_s; j++) {

        if(yi_(j, 1) == 0 || sum(wi_ % yi_.col(1)) == 0) {

          temp(j, 0) = -(0 - exp(y_pi(j))*curr2) * wi_(j) / sum(wi_ % yi_.col(1));
        } else {

          curr2 += 1/curr;
          temp(j, 0) = -(1 - exp(y_pi(j))*curr2) * wi_(j) / sum(wi_ % yi_.col(1));
        }
        curr -= exp(y_pi(j));
      }
      d_a(n_layer) = temp;
      // if(loss_f == "logit")
      //   y_pi = 1 / (1 + exp(-y_pi));
      //
      // if(loss_f == "rmsle") {
      //
      //   y_pi = y_pi % (y_pi > 0);
      //   d_a(n_layer) = -(log(yi_ + 1) - log(y_pi + 1)) /
      //     (y_pi + 1) % y_pi.transform([](double x){return(1*(x > 0));}) % wi_ / sum(wi_);
      // } else if(loss_f == "cross-entropy") {
      //
      //   y_pi = exp(y_pi) / (sum(exp(y_pi), 1) * one_dim_y.t());
      //   d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      // } else {
      //
      //   d_a(n_layer) = -(yi_ - y_pi) % (wi_ * one_dim_y.t()) / sum(wi_);
      // }

      d_w(n_layer) = h(n_layer - 1).t() * d_a(n_layer);
      vec bias_grad = d_a(n_layer).t() * one_sample_size;
      if(learning_rate_adaptive == "momentum") {

        last_dw(n_layer) = last_dw(n_layer) * rho + d_w(n_layer) * learning_rate;
        last_db(n_layer) = last_db(n_layer) * rho + bias_grad    * learning_rate;
        dw(n_layer) = last_dw(n_layer);
        db(n_layer) = last_db(n_layer);
      } else if (learning_rate_adaptive == "adagrad") {

        weight_ss(n_layer) = weight_ss(n_layer) + square(d_w(n_layer));
        bias_ss(n_layer)   = bias_ss(n_layer)   + square(bias_grad);
        dw(n_layer) = d_w(n_layer)/sqrt(weight_ss(n_layer) + epsilon) * learning_rate;
        db(n_layer) = bias_grad   /sqrt(bias_ss(n_layer)   + epsilon) * learning_rate;
      } else if (learning_rate_adaptive == "adadelta") {

        weight_egs(n_layer) = weight_egs(n_layer) * rho + (1-rho) * square(d_w(n_layer));
        bias_egs(n_layer)   = bias_egs(n_layer)   * rho + (1-rho) * square(bias_grad);
        dw(n_layer) = sqrt(weight_es(n_layer) + epsilon) / sqrt(weight_egs(n_layer) + epsilon) % d_w(n_layer);
        db(n_layer) = sqrt(bias_es(n_layer)   + epsilon) / sqrt(bias_egs(n_layer)   + epsilon) % bias_grad;
        weight_es(n_layer) = weight_es(n_layer) * rho + (1-rho) * square(dw(n_layer));
        bias_es(n_layer)   = bias_es(n_layer)   * rho + (1-rho) * square(db(n_layer));
      } else if (learning_rate_adaptive == "adam") {

        adam_ind = adam_ind + 1;
        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        dw(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(n_layer) / (sqrt(vt_w(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
        db(n_layer) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(n_layer) / (sqrt(vt_b(n_layer) / (1-pow(beta2, adam_ind))) + epsilon);
      } else if (learning_rate_adaptive == "amsgrad") {

        mt_w(n_layer) = mt_w(n_layer) * beta1 + (1-beta1) * d_w(n_layer);
        mt_b(n_layer) = mt_b(n_layer) * beta1 + (1-beta1) * bias_grad;
        vt_w(n_layer) = vt_w(n_layer) * beta2 + (1-beta2) * square(d_w(n_layer));
        vt_b(n_layer) = vt_b(n_layer) * beta2 + (1-beta2) * square(bias_grad);
        cube vt_bw(vt_w(n_layer).n_rows, vt_w(n_layer).n_cols, 2);
        mat  vt_bb(vt_b(n_layer).size(), 2);
        vt_bw.slice(0) = vt_w(n_layer);
        vt_bw.slice(1) = vt_bar_w(n_layer);
        vt_bb.col(0) = vt_b(n_layer);
        vt_bb.col(1) = vt_bar_b(n_layer);
        vt_bar_w(n_layer) = max(vt_bw, 2);
        vt_bar_b(n_layer) = max(vt_bb, 1);
        dw(n_layer) = learning_rate * mt_w(n_layer) / (sqrt(vt_bar_w(n_layer)) + epsilon);
        db(n_layer) = learning_rate * mt_b(n_layer) / (sqrt(vt_bar_b(n_layer)) + epsilon);
      } else {

        dw(n_layer) = d_w(n_layer) * learning_rate;
        db(n_layer) = bias_grad    * learning_rate;
      }
      weight_(n_layer) = weight_(n_layer) - dw(n_layer) - l1_reg * (conv_to<mat>::from(weight_(n_layer) > 0) - conv_to<mat>::from(weight_(n_layer) < 0)) - l2_reg * (weight_(n_layer));
      bias_(n_layer)   = bias_(n_layer)   - db(n_layer);
      for(int j = n_layer - 1; j >= 0; j--) {

        d_h(j) = d_a(j + 1) * weight_(j + 1).t();
        if(activ == "sigmoid") {
          d_a(j) = 1/(1+exp(-a(j)));
          d_a(j) = d_h(j) % (d_a(j) % (1-d_a(j)));
        } else if(activ == "tanh") {
          d_a(j) = tanh(a(j));
          d_a(j) = d_h(j) % (1 - square(d_a(j)));
        } else if(activ == "relu") d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0));});
        else if(activ == "prelu")  d_a(j) = d_h(j) % a(j).transform([](double x){return(1*(x > 0) + 0.2*(x <= 0));});
        else if(activ == "elu")    d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));
        else if(activ == "celu")   d_a(j) = d_h(j) % ((a(j) > 0) + (a(j) <= 0) % exp((a(j) <= 0) % a(j)));

        if(j > 0) {
          d_w(j) = h(j - 1).t() * d_a(j);
        } else {
          d_w(j) = xi_.t() * d_a(j);
        }
        vec bias_grad = d_a(j).t() * one_sample_size;
        if(learning_rate_adaptive == "momentum") {

          last_dw(j) = last_dw(j) * rho + d_w(j)    * learning_rate;
          last_db(j) = last_db(j) * rho + bias_grad * learning_rate;
          dw(j) = d_w(j)    * learning_rate + rho * last_dw(j);
          db(j) = bias_grad * learning_rate + rho * last_db(j);
        } else if (learning_rate_adaptive == "adagrad") {

          weight_ss(j) = weight_ss(j) + square(d_w(j));
          bias_ss(j)   = bias_ss(j)   + square(bias_grad);
          dw(j) = d_w(j)   /sqrt(weight_ss(j) + epsilon) * learning_rate;
          db(j) = bias_grad/sqrt(bias_ss(j)   + epsilon) * learning_rate;
        } else if (learning_rate_adaptive == "adadelta") {

          weight_egs(j) = weight_egs(j) * rho + (1-rho) * square(d_w(j));
          bias_egs(j)   = bias_egs(j)   * rho + (1-rho) * square(bias_grad);
          dw(j) = sqrt(weight_es(j) + epsilon) / sqrt(weight_egs(j) + epsilon) % d_w(j);
          db(j) = sqrt(bias_es(j)   + epsilon) / sqrt(bias_egs(j)   + epsilon) % bias_grad;
          weight_es(j) = weight_es(j) * rho + (1-rho) * square(dw(j));
          bias_es(j)   = bias_es(j)   * rho + (1-rho) * square(db(j));
        } else if (learning_rate_adaptive == "adam") {

          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          dw(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(j) / (sqrt(vt_w(j) / (1-pow(beta2, adam_ind))) + epsilon);
          db(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(j) / (sqrt(vt_b(j) / (1-pow(beta2, adam_ind))) + epsilon);
        } else if (learning_rate_adaptive == "amsgrad") {

          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          cube vt_bw(vt_w(j).n_rows, vt_w(j).n_cols, 2);
          mat  vt_bb(vt_b(j).size(), 2);
          vt_bw.slice(0) = vt_w(j);
          vt_bw.slice(1) = vt_bar_w(j);
          vt_bb.col(0) = vt_b(j);
          vt_bb.col(1) = vt_bar_b(j);
          vt_bar_w(j) = max(vt_bw, 2);
          vt_bar_b(j) = max(vt_bb, 1);
          dw(j) = learning_rate * mt_w(j) / (sqrt(vt_bar_w(j)) + epsilon);
          db(j) = learning_rate * mt_b(j) / (sqrt(vt_bar_b(j)) + epsilon);
        } else {

          dw(j) = d_w(j)    * learning_rate;
          db(j) = bias_grad * learning_rate;
        }
        weight_(j) = weight_(j) - dw(j) - l1_reg * (conv_to<mat>::from(weight_(j) > 0) - conv_to<mat>::from(weight_(j) < 0)) - l2_reg * (weight_(j));
        bias_(j)   = bias_(j)   - db(j);  // column mean
      }
    }

    if(valid) {

      mat pred;
      int n_s = x_valid_.n_rows;
      vec one_sample_size = rep(1.0, n_s);
      vec one_dim_y = rep(1.0, y_.n_cols);
      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {
          pred = x_valid_ * weight_(j) + one_sample_size * bias_(j).t();
        } else {
          pred = pred * weight_(j) + one_sample_size * bias_(j).t();
        }
        if(activ == "sigmoid")    pred = 1/(1+exp(-pred));
        else if(activ == "tanh")  pred = tanh(pred);
        else if(activ == "relu")  pred = pred % (pred > 0);
        else if(activ == "prelu") pred = pred % (pred > 0) + (pred <= 0) % pred*0.2;
        else if(activ == "elu")   pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
        else if(activ == "celu")  pred = pred % (pred > 0) + exp((pred <= 0) % pred) - 1;
      }
      mat y_pred = pred * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      mat temp(n_s, 1);
      double curr = sum(sum(exp(y_pred)));
      for(int j = 0; j < n_s; j++) {

        if(y_valid_(j, 1) == 0 || sum(w_valid_ % y_valid_.col(1)) == 0) {

          temp(j, 0) = 0;
        } else {

          temp(j, 0) = -(y_pred(j) - log(curr)) * w_valid_(j) / sum(w_valid_ % y_valid_.col(1));
        }
        curr -= exp(y_pred(j));
      }
      loss[k] = sum(sum(temp));
      // if(loss_f == "logit") {
      //
      //   y_pred = 1 / (1 + exp(-y_pred));
      //   loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred) + (1-y_valid_) % log(1-y_pred), 1)) / sum(w_valid_);
      // } else if(loss_f == "mse") {
      //
      //   loss[k] = sum(w_valid_ % sum(pow(y_valid_ - y_pred, 2), 1)) / sum(w_valid_);
      // } else if(loss_f == "cross-entropy") {
      //
      //   y_pred = exp(y_pred) / (sum(exp(y_pred), 1) * one_dim_y.t());
      //   loss[k] = -sum(w_valid_ % sum(y_valid_ % log(y_pred), 1)) / sum(w_valid_);
      // } else if(loss_f == "rmsle") {
      //
      //   y_pred = y_pred % (y_pred > 0);
      //   loss[k] = sum(w_valid_ % pow(log(y_valid_ + 1) - log(y_pred + 1), 2)) / sum(w_valid_);
      // }

      if(!is_finite(loss[k])) {

        break_k = k-1;
        break;
      } else {

        if(loss[k] < best_loss) {

          best_loss = loss[k];
          best_weight = weight_;
          best_bias = bias_;
        }

        if(k > early_stop_det) {
          if(prod(loss.subvec(k-early_stop_det+1, k) > loss.subvec(k-early_stop_det, k-1)) > 0) {

            break_k = k;
            break;
          }
        }
      }
    }
  }

  List best_weight_(best_weight.size());
  List best_bias_(best_weight.size());
  if(early_stop) {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(best_weight(i));
      best_bias_(i) = wrap(best_bias(i).t());
    }
  } else {

    for(unsigned int i = 0; i < best_weight.size(); i++) {

      best_weight_(i) = wrap(weight_(i));
      best_bias_(i) = wrap(bias_(i).t());
    }
  }

  List result(4);
  result(0) = best_weight_;
  result(1) = best_bias_;
  result(2) = loss;
  result(3) = break_k;
  return(result);
}


