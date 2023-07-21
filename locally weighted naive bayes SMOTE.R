locally_weighted_naive_bayes_SMOTE <- function(x_train, y_train, k_neighbour = 50){
  
  n_train <- nrow(x_train)
  
  class_names <- unique(y_train)
  k_classes <- length(class_names)
  
  class_neg <- names(which.max(table(y_train)))
  class_pos <- as.character(class_names[class_names != class_neg])
  
  n_neg <- sum(y_train == class_neg)
  n_pos <- sum(y_train == class_pos)
  imb_rate <- n_neg/n_pos
  
  n_train <- nrow(x_train)
  n_classes <- sapply(class_names, function(m) sum(y_train == m))
  
  priors_normal <- n_classes/n_train

  return(list(n_train = n_train,
              x_train = x_train,
              y_train = y_train,
              n_classes = n_classes,
              k_classes = k_classes,
              class_names = class_names,
              class_pos = class_pos,
              class_neg = class_neg,
              imb_rate = imb_rate,
              k_neighbour = k_neighbour))
}

predict_locally_weighted_naive_bayes_SMOTE <- function(object, newdata, type = "prob"){
  n_train <- object$n_train
  x_train <- object$x_train
  y_train <- object$y_train
  n_classes <- object$n_classes
  k_classes <- object$k_classes
  class_names <- object$class_names
  class_pos <- object$class_pos
  class_neg <- object$class_neg
  imb_rate <- object$imb_rate
  k_neighbour <- object$k_neighbour
  
  x_test <- newdata
  n_test <- nrow(x_test)
  
  posteriors_all <- matrix(NA, nrow = n_test, ncol = k_classes)
  
  for (i in 1:n_test) {
    x_selected <- x_test[i,,drop = FALSE]
    
    # nb_dists <- FNN::knnx.dist(data = x_train, x_selected, k = n_train)
    # nb_index <- FNN::knnx.index(data = x_train, x_selected, k = n_train)
    
    m_nb <- RANN::nn2(data = as.matrix(x_train), query = as.matrix(x_selected), k = pmin(k_neighbour, n_train))
    nb_dists <- m_nb$nn.dists
    nb_index <- m_nb$nn.idx
    
    x_train_selected <- x_train[nb_index,]
    y_train_selected <- y_train[nb_index]
    
    if (all(table(y_train_selected) > 2)) {
      dat_SMOTE <- SMOTE(x_train = x_train_selected, 
                         y_train = y_train_selected, 
                         k_neighbour = 2, 
                         imb_rate = imb_rate, 
                         class_neg = as.character(class_neg), 
                         class_pos = as.character(class_pos))
      x_train_selected_SMOTE <- dat_SMOTE$x_new
      y_train_selected_SMOTE <- dat_SMOTE$y_new
      n_SMOTE <- nrow(x_train_selected_SMOTE)
    } else {
      x_train_selected_SMOTE <- x_train_selected
      y_train_selected_SMOTE <- y_train_selected
      n_SMOTE <- nrow(x_train_selected_SMOTE)
    }

    m_nb <- RANN::nn2(data = as.matrix(x_train_selected_SMOTE), query = as.matrix(x_selected), k = n_SMOTE)
    nb_dists <- m_nb$nn.dists
    nb_index <- m_nb$nn.idx

    weights <- nb_dists/max(nb_dists)

    # weights <- numeric(n_train)
    # weights[nb_index[2:(k_neighbour + 1)]] <- 1 - nb_dists[2:(k_neighbour + 1)]/nb_dists[k_neighbour + 1]
    # 
    weights <- weights*k_neighbour/sum(weights) ### niyesini anlamadım. ağırlıkların toplamını k yapıyor.
    x_train_selected_classes <- lapply(class_names, function(m) {
      x_train_selected_SMOTE[y_train_selected_SMOTE == m,,drop = FALSE]
      })
    
    weights_classes <- lapply(class_names, function(m) weights[y_train_selected_SMOTE == m])
    
    means <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
      ww <- weights_classes[[m2]]/sum(weights_classes[[m2]])*n_test
      Hmisc::wtd.mean(x = x_train_selected_classes[[m2]][,m,drop = FALSE], weights = ww, na.rm = TRUE)
    }))
    
    stds <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
      ww <- weights_classes[[m2]]/sum(weights_classes[[m2]])*n_test
      sdsd <- sqrt(Hmisc::wtd.var(x = x_train_selected_classes[[m2]][,m,drop = FALSE], weights = ww, na.rm = TRUE))
      sdsd[sdsd == 0] <- 1e-20
      return(sdsd)
    }))
    
    priors <- sapply(weights_classes, sum)/sum(weights)
    likelihoods <- t(sapply(1:k_classes, function(m) dnorm(x = unlist(x_selected), mean = means[[m]], sd = stds[[m]])))
    
    priors[is.nan(priors)] <- 1e-20
    priors[is.na(priors)] <- 1e-20
    priors[is.infinite(priors)] <- 1e100-20
    priors <- priors/sum(priors)
    
    likelihoods[is.nan(likelihoods)] <- 1e-20
    likelihoods[is.na(likelihoods)] <- 1e-20
    likelihoods[is.infinite(likelihoods)] <- 1e100
    
    posteriors <- apply(cbind(priors, likelihoods), 1, prod)
    if(all(posteriors == 0)){
      posteriors <- runif(length(posteriors), min = 0.49, max = 0.51)
    }
    posteriors[is.infinite(posteriors)] <- .Machine$double.xmax
    posteriors <- posteriors/sum(posteriors)
    
    
    posteriors_all[i,] <- posteriors
  }
  
  colnames(posteriors_all) <- class_names
  
  if (type == "prob") {
    return(posteriors_all)
  }
  if (type == "pred") {
    predictions <- apply(posteriors_all, 1, function(m) class_names[which.max(m)])
    return(predictions)
  }
}
