#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
SEXP backprop(NumericVector n_hidden, double w_ini, // List weight, List bias,
              bool load_param, List weight, List bias,
              NumericMatrix x, NumericVector y, NumericVector w, bool valid,
              NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid,
              std::string activ,
              int n_epoch, int n_batch, std::string model_type,
              double learning_rate, double l1_reg, double l2_reg, bool early_stop, int early_stop_det,
              std::string learning_rate_adaptive, double rho, double epsilon, double beta1, double beta2,
              std::string loss_f) {

  mat x_ = as<mat>(x);
  vec y_ = as<vec>(y);
  vec w_ = as<vec>(w);
  mat x_valid_;
  vec y_valid_;
  vec w_valid_;

  if(valid) {

    x_valid_ = as<mat>(x_valid);
    y_valid_ = as<vec>(y_valid);
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
  //momentum
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
      dw(i) = mat(n_hidden[i-1], 1, fill::zeros);
      db(i) = vec(1, fill::zeros);
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
    y_ = y_.elem(new_order);
    w_ = w_.elem(new_order);

    for(int i = 0; i < n_round; i++) {

      mat xi_ = x_.rows(i_bgn[i], i_end[i]);
      vec yi_ = y_.subvec(i_bgn[i], i_end[i]);
      vec wi_ = w_.subvec(i_bgn[i], i_end[i]);
      int n_s = xi_.n_rows;
      vec one_sample_size = rep(1.0, n_s);

      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {

          a(j) = xi_ * weight_(j) + one_sample_size * bias_(j).t();
          // h(j) = as<mat>(activate(a(j)));
        } else{

          a(j) = h(j-1) * weight_(j) + one_sample_size * bias_(j).t();
          // h(j) = as<mat>(activate(a(j)));
        }
        if(activ == "sigmoid")    h(j) = 1/(1+exp(-a(j)));
        else if(activ == "tanh")  h(j) = tanh(a(j));
        else if(activ == "relu")  h(j) = a(j) % (a(j) > 0);
        else if(activ == "prelu") h(j) = a(j) % (a(j) > 0) + (a(j) <= 0) % a(j)*0.2;
        else if(activ == "elu")   h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
        else if(activ == "celu")  h(j) = a(j) % (a(j) > 0) + exp((a(j) <= 0) % a(j)) - 1;
      }
      vec y_pi = h(n_layer - 1) * weight_(n_layer) + one_sample_size * bias_(n_layer);
      //if(model_type == "classification")
      if(loss_f == "logit")
        y_pi = 1 / (1 + exp(-y_pi));
      if(loss_f == "rmsle") {

        y_pi = y_pi % (y_pi > 0);
        d_a(n_layer) = -(log(yi_ + 1) - log(y_pi + 1)) /
          (y_pi + 1) % y_pi.transform([](double x){return(1*(x > 0));}) % wi_ / sum(wi_);
      } else {

        d_a(n_layer) = -(yi_ - y_pi) % wi_ / sum(wi_);
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
      } else {

        dw(n_layer) = d_w(n_layer) * learning_rate;
        db(n_layer) = bias_grad    * learning_rate;
      }
      weight_(n_layer) = weight_(n_layer) - dw(n_layer) - l1_reg * (conv_to<mat>::from(weight_(n_layer) > 0) - conv_to<mat>::from(weight_(n_layer) < 0)) - l2_reg * (weight_(n_layer));
      bias_(n_layer)   = bias_(n_layer)   - db(n_layer);
      for(int j = n_layer - 1; j >= 0; j--) {

        d_h(j) = d_a(j + 1) * weight_(j + 1).t();
        // d_a(j) = d_h(j) % as<mat>(activate_(a(j)));
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

          adam_ind ++;
          mt_w(j) = mt_w(j) * beta1 + (1-beta1) * d_w(j);
          mt_b(j) = mt_b(j) * beta1 + (1-beta1) * bias_grad;
          vt_w(j) = vt_w(j) * beta2 + (1-beta2) * square(d_w(j));
          vt_b(j) = vt_b(j) * beta2 + (1-beta2) * square(bias_grad);
          dw(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_w(j) / (sqrt(vt_w(j) / (1-pow(beta2, adam_ind))) + epsilon);
          db(j) = learning_rate / (1-pow(beta1, adam_ind)) * mt_b(j) / (sqrt(vt_b(j) / (1-pow(beta2, adam_ind))) + epsilon);
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
      vec y_pred(n_s);
      vec one_sample_size = rep(1.0, n_s);
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
      y_pred = pred * weight_(n_layer) + one_sample_size * bias_(n_layer).t();
      //if(model_type == "classification") {
      if(loss_f == "logit") {

        y_pred = 1 / (1 + exp(-y_pred));
        loss[k] = -sum(w_valid_ % (y_valid_ % log(y_pred) + (1-y_valid_) % log(1-y_pred))) / sum(w_valid_);
      } else if(loss_f == "mse") {

        loss[k] = sum(w_valid_ % pow(y_valid_ - y_pred, 2)) / sum(w_valid_);
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

  // return(List::create(Rcpp::Named("weight") = best_weight_,
  //                     Rcpp::Named("bias") = best_bias_,
  //                     Rcpp::Named("loss") = loss));
}

