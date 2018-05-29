
// Latent factor model for chemical abundances from multiple studies

data {
  int<lower=1> N; // number of stars
  int<lower=1> D; // dimensionality of the data (number of labels)
  int<lower=1> M; // number of surveys (or studies)
  vector[D] y[N, M]; // the labels as reported by various surveys.
  vector<lower=0>[D] scales; // fixed relative scales for latent factors
}

/*
transformed data {
  // todo: remove means from the data rather than assuming the user did
  vector[D] mu; // mean values of the data.
  mu = rep_vector(0.0, D);
}
*/

parameters {
  vector[D] X[N]; // latent factors for each star
  vector[D] theta[M]; // survey transformations

  vector<lower=0>[D] psi; // intrinsic variance in labels
  vector<lower=0>[M] phi[D]; // variance on survey labels
}

/*
transformed parameters {
  cov_matrix[D] Sigma[M];
  // TODO: this is inefficient (for indexing, ram, etc)
  for (m in 1:M)
    for (d in 1:D)
      Sigma[d, m] = phi[m, d] + psi[d];
}
*/

model {
  for (d in 1:D) {
    // TODO: Check that Stan is vectorizing this as we expect!
    // todo: improve this shit
    X[:, d] ~ normal(rep_vector(0, N), rep_vector(scales[d], N));
    theta[:, d] ~ normal(rep_vector(0, M), rep_vector(1, M));
    psi[d] ~ normal(0, 1);
    phi[d, :] ~ normal(rep_vector(0, M), rep_vector(1, M));
  }

  // so inefficient
  for (n in 1:N) {
    for (m in 1:M) {
      //print("n m mu", n, m, mu);
      y[n, m] ~ normal(rows_dot_product(X[n], theta[m]),
                       sqrt(to_vector(phi[:, m]) + psi));
    }
  }
}