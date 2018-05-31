
/*
  Latent factor model for chemical abundances from multiple studies, allowing
  for missing data.
*/

data {
  int<lower=1> N; // number of stars
  int<lower=1> D; // dimensionality of the data (number of labels)
  int<lower=1> M; // number of surveys (or studies)
  vector[D] y[N, M]; // the labels as reported by various surveys.
  vector<lower=0>[D] scales; // fixed relative scales for latent factors
  vector[D] y_additive_variance[N, M]; // variance to add for missing data.
}

transformed data {
  vector[D] mu; // the mean of the data in each dimension
  int<lower=1> Q; // the number of non-zero lower-triangular entries that we
                  // need for the decomposoition of our theta matrix
  Q = M * choose(D, 2);

  // TODO: Stop assuming that the user is not an idiot
  mu = rep_vector(0.0, D);
}

parameters {
  vector[D] X[N]; // latent factors for each star
  vector<lower=0>[M] phi[D]; // variance on survey labels

  vector[Q] L_lower_triangular; // lower triangular entries of the decomposition
                                // of the  theta matrix
  vector<lower=0>[M] L_diag[D]; // diagonal entries of the decomposition of the 
                                // theta matrix
}

transformed parameters {
  cholesky_factor_cov[D, D] L[M];
  matrix[D, D] theta[M];
  {
    int q = 0;

    for (m in 1:M)
      for (i in 1:D)
        for (j in (i + 1):D) 
          L[m, i, j] = 0.0;

    for (m in 1:M) {
      for (i in 1:D) {
        L[m, i, i] = L_diag[i, m];
        for (j in (i + 1):D) {
          q = q + 1;
          L[m, j, i] = L_lower_triangular[q];
        }
      }
    }

    for (m in 1:M)
      theta[m] = multiply_lower_tri_self_transpose(L[m]);
  }
}

model {

  // Place priors on various properties.
  for (d in 1:D) {
    X[:, d] ~ normal(rep_vector(0, N), rep_vector(scales[d], N));
    phi[d] ~ normal(rep_vector(0, M), rep_vector(1, M));

    // TODO: Should we be placing an inverted Wishart prior or something on this
    //       shit?
    //L_lower_triangular

    // TODO: revisit this prior
    L_diag[d] ~ normal(rep_vector(1, M), rep_vector(0.1, M));
  }

  for (n in 1:N) 
    for (m in 1:M)
      y[n, m, :] ~ normal(to_row_vector(X[n]) * theta[m], 
                          sqrt(to_vector(phi[:, m]) + y_additive_variance[n, m, :]));
}

