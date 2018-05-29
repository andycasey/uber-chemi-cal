
// Latent factor model for chemical abundances from multiple studies

data {
  int<lower=1> N; // number of stars
  int<lower=1> D; // dimensionality of the data (number of labels)
  int<lower=1> M; // number of surveys (or studies)
  vector[D] y[N, M]; // the labels as reported by various surveys.
  vector[D] scales; // the fixed scales for different labels
}


transformed data {
  // todo: remove means from the data rather than assuming the user did
  vector[D] mu; // mean values of the data.
  mu = rep_vector(0.0, D);
}

parameters {
  vector[D] X[N]; 
  vector[D] theta[M];

  vector<lower=0>[D] psi; // intrinsic variance in labels
  vector<lower=0>[M, D] phi; // variance on survey labels
}

transformed parameters {
  cov_matrix[D] Sigma[M];
  // TODO: this is inefficient (for indexing, ram, etc)
  for (m in 1:M)
    for (d in 1:D)
      Sigma[d, m] = phi[m, d] + psi[d];
}

model {
  // TODO: Check that Stan is vectorizing this as we expect!
  theta ~ normal(0, 1);

  for (i in 1:D) {
    // TODO: Check that Stan is vectorizing this as we expect!
    X[d] ~ normal(rep_vector(0, N), rep_vector(scales[d], N));
  }

  // this is wrong
  y ~ multi_normal(X * theta - mu, Sigma);
}