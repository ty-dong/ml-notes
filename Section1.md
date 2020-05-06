# Chapter 1: The Machine Learning Landscape

## whether or not the system can learn incrementally

- batch learning: If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset
- out-of-core learning: train systems on huge datasets that cannot fit in one machine’s main memory
- online learning: 
  - train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches.
  - need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance.

## how model generalize

- instance-based learning: the system learns the examples by heart, then generalizes to new cases by comparing them to the learned examples (or a subset of them), using a similarity measure.
- model-based learning: build a model of these examples, then use that model to make predictions. 

## validate models

- it is common to use 80% of the data for training and hold out 20% for testing. However, this depends on the size of the dataset: if it contains 10 million instances, then holding out 1% means your test set will contain 100,000 instances: that’s probably more than enough to get a good estimate of the generalization error.
- holdout validation: you simply hold out part of the training set to evaluate several candidate models and select the best one. The new heldout set is called the validation set
- the most important rule to remember is that the validation set and the test must be **as representative as possible** of the data you expect to use in production 

## challenges of machine learning

- Insufficient Quantity of Training Data
- Nonrepresentative Training Data
- Poor-Quality Data
- Irrelevant Features
- Overfitting the Training Data
- Underfitting the Training Data



# Chapter 2: End-to-End Machine Learning Project

## transformers and pipelines

- it is important to fit the scalers to the training data only, not to the full dataset
- All but the last estimator must be transformers

e.g.

`old_num_pipeline = Pipeline([`
        `('selector', OldDataFrameSelector(num_attribs)),`
        `('imputer', SimpleImputer(strategy="median")),`
        `('attribs_adder', CombinedAttributesAdder()),`  (add new attributes)
        `('std_scaler', StandardScaler()),`
    `])`

`old_cat_pipeline = Pipeline([`
        `('selector', OldDataFrameSelector(cat_attribs)),`
        `('cat_encoder', OneHotEncoder(sparse=False)),`
    `])`

`
old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])
`

## save models

`from sklearn.externals import joblib`
`joblib.dump(my_model, "my_model.pkl") # and later...`
`my_model_loaded = joblib.load("my_model.pkl")`

## Fine tuning the model

- If GridSearchCV is initialized with refit=True (which is the default), then once it finds the best estimator using crossvalidation, it retrains it on the whole training set. This is usually a good idea since feeding it more data will likely improve its performance.
- The performance will usually be slightly worse than what you measured using crossvalidation if you did a lot of hyperparameter tuning (because your system ends up fine-tuned to perform well on the validation data, and will likely not perform as well on unknown datasets).



# Chapter 3: Classification

## Preformance measures

- This demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets
- confusion matrix: Each row in a confusion matrix represents an actual class, while each column represents a predicted class
- recall can only go down when the threshold is increased
- ROC curve plots sensitivity (recall) versus 1 – specificity (specificity = true negative rate, which is the ratio of negative instances that are correctly classified as negative).
- Since the ROC curve is so similar to the precision/recall (or PR) curve, you may wonder how to decide which one to use. As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives, and the ROC curve otherwise.

## Multiclass Classification

- One-versus-all (OvA) strategy classifies whether a sample belongs to one class or not. Another strategy is to train a binary classifier for every pair of digits: one to distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. This is called the **one-versus-one** (OvO) strategy. If there are N classes, you need to train N × (N – 1) / 2 classifiers. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish.



# Chapter 4: Training Models

## Gradient descent

- When using Gradient Descent, you should ensure that all features have a similar scale (e.g., using Scikit-Learn’s StandardScaler class), or else it will take much longer to converge.

### Batch gradient descent

when the cost function is convex and its slope does not change abruptly (as is the case for the MSE cost function), Batch Gradient Descent with a fixed learning rate will eventually converge to the optimal solution, but you may have to wait a while: it can take $O(1/ \epsilon)$ iterations to reach the optimum within a range of $\epsilon$ depending on the shape of the cost function.

### Stochastic gradient descent

- randomness is good to escape from local optima, but bad because it means that the algorithm can never settle at the minimum. One solution to this dilemma is to gradually reduce the learning rate. The steps start out large (which helps make quick progress and escape local minima), then get smaller and smaller, allowing the algorithm to settle at the global minimum.
- The function that determines the learning rate at each iteration is called the **learning schedule**. If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up frozen halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early.

- The main advantage of Mini-batch GD over Stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.
- By convention we iterate by rounds of m iterations; each round is called an **epoch**.

## Learning curves

- bias: this part of the generalization error is due to wrong assumptions
- variance: This part is due to the model’s excessive sensitivity to small variations in the training data.
- irreducible error: this part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data

## Regularized linear models

- **ridge regression**: it is important to scale the data (e.g., using a StandardScaler) before performing Ridge Regression, as it is sensitive to the scale of the input features. This is true of most regularized models.
- **lasso regression**: an important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features
- **Elastic Net** is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge Regression, and when r = 1, it is equivalent to Lasso Regression $$J(\theta)=MSE(\theta)+r\alpha\sum_{i=1}^n|\theta_i|+\frac{1-r}{2}\alpha\sum_{i=1}^n\theta_i^2$$
- **Early stop**: with Stochastic and Mini-batch Gradient Descent, the curves are not so smooth, and it may be hard to know whether you have reached the minimum or not. One solution is to stop only after the validation error has been above the minimum for some time (when you are confident that the model will not do any better), then roll back the model parameters to the point where the validation error was at a minimum.
  `sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)`
  with warm_start=True, when the fit() method is called, it just continues training where it left off instead of restarting from scratch.
- It is important to scale the data (e.g., using a StandardScaler) before performing Ridge Regression, as it is sensitive to the scale of the input features. This is true of most regularized models.

## Logistic Regression

- estimated probability: $\hat{p} = \sigma(x^T\theta)$ where logistic function $\sigma(t) = \frac{1}{1+\exp(-t)}$
- logistic regression model prediction: $\hat{y} = 0$ if $\hat{p}<0.5$ else $1$

- cost function $c(\theta)=-\log(\hat{p})$ if $y=1$ else $-\log(1-\hat{p})$ if $y=0$

## Softmax Regression

- The Softmax Regression classifier predicts **only one class at a time** (i.e., it is multiclass, not multioutput) so it should be used only with mutually exclusive classes such as different types of plants. You cannot use it to recognize multiple people in one picture.


# Chapter 5 SVM

## Linear SVM Classification

- think of an SVM classifier as fitting the widest possible street (represented by the parallel dashed lines) between the classes. This is called **large margin classification**.
- instances located on the edge of the street are called the **support vectors**
- SVMs are sensitive to the feature scales
- two main issues with hard margin classification：
  - it only works if the data is linearly separable
  - it is quite sensitive to outliers. 
- **soft margin classification**: find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side).
- If SVM model is overfitting, you can try regularizing it by reducing C. On the left, using a low C value the margin is quite large, but many instances end up on the street. On the right, using a high C value the classifier makes fewer margin violations but ends up with a smaller margin.
- Besides the LinearSVC class, you could use the SVC class, using `SVC(kernel="linear", C=1)`, but it is much slower, especially with large training sets, so it is not recommended. Another option is to use the SGDClassifier class, with `SGDClassifier(loss="hinge", alpha=1/(m*C))`. This applies regular Stochastic Gradient Descent (see Chapter 4) to train a linear SVM classifier. It does not converge as fast as the LinearSVC class, but it can be useful to handle huge datasets that do not fit in memory (out-of-core training), or to handle online classification tasks.
- The LinearSVC class **regularizes the bias term**, so you should center the training set first by subtracting its mean. This is automatic if you scale the data using the StandardScaler. Moreover, make sure you set the loss hyperparameter to "hinge", as it is not the default value. Finally, for better performance you should set the dual hyperparameter to False, unless there are more features than training instances

## Nonlinear SVM Classification

- One approach to handling nonlinear datasets is to add more features, such as polynomial features
- Adding polynomial features is simple to implement and can work great with all sorts of Machine Learning algorithms (not just SVMs), but at a low polynomial degree it cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow.
- Parameter in SVC: the **hyperparameter coef0** controls how much the model is influenced by high-degree polynomials versus low-degree polynomials.
- A common approach to find the right hyperparameter values is to use **grid search**. It is often faster to first do a very coarse grid search, then a finer grid search around the best values found.

### Adding Similarity Features

- **similarity function**: measures how much each instance resembles a particular landmark.
- Gaussian radial basis function: $\phi_{\gamma}(x, l)=\exp(-\gamma||x-l||^2)$
- how to choose landmark? You may wonder how to select the landmarks. The simplest approach is to **create a landmark at the location of each and every instance in the dataset**. This creates many dimensions and thus increases the chances that the transformed training set will be linearly separable. The downside is that a training set with m instances and n features gets transformed into a training set with m instances and m features (assuming you drop the original features). If your training set is very large, you end up with an equally large number of features.



- Gaussian RBF kernel: increasing gamma makes the bell-shape curve narrower, and as a result each instance’s range of influence is smaller: the decision boundary ends up being more irregular, wiggling around individual instances. Conversely, a small gamma value makes the bell-shaped curve wider, so instances have a larger range of influence, and the decision boundary ends up smoother.
- $\gamma$ acts like a regularization hyperparameter: if your model is overfitting, you should reduce it, and if it is underfitting, you should increase it (similar to the $C$ hyperparameter).
- how can you decide which one to use? 
  - As a rule of thumb, you should always try the linear kernel first (remember that LinearSVC is much faster than SVC(kernel="linear")), especially if the training set is very large or if it has plenty of features. 
  - If the training set is not too large, you should try the Gaussian RBF kernel as well; it works well in most cases. 
  - Then if you have spare time and computing power, you can also experiment with a few other kernels using cross-validation and grid search, especially if there are kernels specialized for your training set’s data structure.

| Class         | Time complexity      | Out-of-core support | Scaling required | Kernel trick |
| ------------- | -------------------- | ------------------- | ---------------- | ------------ |
| LinearSVC     | O(m*n)               | no                  | yes              | no           |
| SDGClassifier | O(m*n)               | yes                 | yes              | no           |
| SVC           | O(m^2*n) to O(m^3*n) | no                  | yes              | yes          |

## SVM Regression

- instead of trying to fit the largest possible street between two classes while limiting margin violations, SVM Regression tries to **fit as many instances as possible** on the street while limiting margin violations
- The width of the street is controlled by a hyperparameter $\epsilon$, large $\epsilon$ leads to large margins.
- Adding more training instances within the margin does not affect the model’s predictions; thus, the model is said to be ϵ-insensitive.

## Under the hood

- one way to train a hard margin linear SVM classifier is just to use an off-the-shelf Quadratic Programming  solver
- The dual problem is faster to solve than the primal when the number of training instances is smaller than the number of features.

### Kernel trick

- the essence of the kernel trick: the whole process much more computationally efficient.
- According to **Mercer’s theorem**, if a function $K(a, b)$ respects a few mathematical conditions called Mercer’s conditions ($K$ must be continuous, symmetric in its arguments so $K(a, b) = K(b, a)$, etc.), then there exists a function $\phi$ that maps a and b into another space (possibly with much higher dimensions) such that $K(a, b) = \phi(a)^T\phi(b)$. So you can use K as a kernel since you know $\phi$ exists, even if you don’t know what $\phi$ is.
- Note that some frequently used kernels (such as the Sigmoid kernel) don’t respect all of Mercer’s conditions, yet they generally work well in practice.


# Chapter 6 Decision Tree

- One of the many qualities of Decision Trees is that they require very little data preparation. In particular, they don’t require feature scaling or centering at all.
- attributes of decision tree's nodes:
  - samples: how many training instances it applies to
  - value: how many training instances of each class this node applies to
  - gini: measures its impurity
    Def: $G_i=1-\sum_{k=1}^np_{i,k}^2$ where $p_{i,k}$ is the ratio of class k instances among the training instances in the i-th node.

- Scikit-Learn uses the CART algorithm, which produces only binary trees

## The CART Training Algorithm

Classification And Regression Tree alg: 

- split the training set in two subsets using a single feature $k$ and a threshold $t_k$. The pair $(k. t_k)$ that produces the purest subsets. 
  The cost function that the alg tries to minimize is given by: 
  $$J(k, t_k)=\frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}$$ where $G_{left}$ measures the impurity of the left subset and $m_{left}$ is the number of instances in the left dataset.
- it splits the subsets using the same logic, then the sub-subsets and so on, recursively. It stops recursing once it reaches the maximum depth (defined by the `max_depth` hyperparameter), or if it cannot find a split that will reduce impurity.
  A few other hyperparameters (described in a moment) control additional stopping conditions (`min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, and `max_leaf_nodes`).

CART algorithm is a **greedy** algorithm. A greedy algorithm often produces a reasonably good solution, but it is not guaranteed to be the optimal solution. Finding the optimal tree is known to be an NP-Complete problem: it requires $O(\exp(m))$ time,

## Computational complexity

$m$: number of instances.
$n$: number of features

- Making predictions requires traversing the Decision Tree from the root to a leaf. The overall prediction complexity is just $O(\log_2 (m))$. So predictions are very fast, even when dealing with large training sets.
- The training algorithm compares all features (or less if `max_features` is set) on all samples at each node. This results in a training complexity of $O(n\times m\log(m))$.
- Scikit-Learn can speed up training by presorting the data (set `presort=True`), but this slows down training considerably for larger training sets.

## Gini or entropy?

Entropy: $H_i =-\sum_{k=1, p_{i,k}\neq0}^np_{i,k}\log_2(p_{i,k})$

The truth is, most of the time it does not make a big difference: they lead to similar trees. Gini impurity is slightly faster to compute, so it is a good default. However, when they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees. 

## Regularization parameters

- Decision Trees make very few assumptions about the training data. If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely, and most likely overfitting it.
- parameters that can be used to regularize parameters:
  - `max_depth`
  - `min_samples_split`: the minimum number of samples a node must have before it can be split
  - `min_samples_leaf`: the minimum number of samples a leaf node must have
  - `min_weight_fraction_leaf`: same as min_samples_leaf but expressed as a fraction of the total number of weighted Decision Trees instances
  - `max_leaf_nodes`: maximum number of leaf nodes
  - `max_features`: maximum number of features that are evaluated for splitting at each node

## Regression

The CART algorithm works mostly the same way as earlier, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE.

## Limitations

- Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation.
- the main issue with Decision Trees is that they are very sensitive to small variations in the training data.



# Chapter 7 Ensemble Learning and Random Forests

- A group of predictors is called an **ensemble**

## Voting classes

- A very simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes. This majority-vote classifier is called a **hard voting classifier**
- Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy.
- you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers. This is called **soft voting**. It often achieves higher performance than hard voting because it gives more weight to highly confident votes.

## Bagging and pasting

- Another approach is to use the same training algorithm for every predictor, but to train them on different random subsets of the training set. When sampling is performed with replacement, this method is called **bagging** (short for bootstrap aggregating). When sampling is performed without replacement, it is called **pasting**
- Each predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance
- Predictors can all be trained in parallel. Also, predictions can be made in parallel.
- The BaggingClassifier automatically performs soft voting instead of hard voting if the base classifier can estimate class probabilities
- Bootstrapping introduces a bit **more diversity** in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting, but this also means that predictors end up being less correlated so the ensemble’s variance is reduced. **Overall, bagging often results in better models**

### Out-of-bag evaluation

- By default a BaggingClassifier samples $m$ training instances with replacement (`bootstrap=True`), where $m$ is the size of the training set. This means that only about 63% of the training instances are sampled on average for each predictor. The remaining 37% of the training instances that are not sampled are called **out-of-bag (oob)** instances. Note that they are not the same 37% for all predictors. As $m$ grows, this ratio approaches $1 – \exp(–1) ≈ 63.212%$.
- Since a predictor never sees the oob instances during training, it can be evaluated on these instances, without the need for a separate validation set. You can evaluate the ensemble itself by averaging out the oob evaluations of each predictor.
- you can set `oob_score=True` when creating a BaggingClassifier to request an automatic oob evaluation after training.
- The BaggingClassifier class supports sampling the features as well. This is controlled by two hyperparameters: `max_features` and `bootstrap_features`. 
- Sampling both training instances and features is called the **Random Patches** method. Keeping all training instances but sampling features is called the **Random Subspaces** method. 

## Random forest

### Extra trees

- It is possible to make trees even more random by also using random thresholds for each feature rather than searching for the best possible thresholds
- A forest of such extremely random trees is simply called an **Extremely Randomized Trees** ensemble. This trades more bias for a lower variance.
- It is hard to tell in advance whether a RandomForestClassifier will perform better or worse than an ExtraTreesClassifier. Generally, the only way to know is to try both and compare them using cross-validation

### Feature importance

- Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest). It is a weighted average, where each node’s weight is equal to the number of training samples that are associated with it
- You can access the result using the `feature_importances_` variable.

## Boosting

The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

### AdaBoost

- pay a bit more attention to the training instances that the predecessor underfitted.

- There is one important **drawback** to this sequential learning technique: it cannot be parallelized (or only partially), since each predictor can only be trained after the previous predictor has been trained and evaluated. As a result, it does not scale as well as bagging or pasting.

- Alg

  - Each instance weight $w^{(i)}$ is initially set to $\frac1m$ .

  - A first predictor is trained and its weighted error rate $r_1$ is computed on the training set. Weighted error rate of $j^{\text{th}}$ predictor is: $$r_j=\frac{\sum\limits_{i=1,\hat{y}_j^{(i)}\neq y^{(i)}}^mw^{(i)}}{\sum_{i=1}^mw^{(i)}}$$

    where $\hat{y}_j^{(i)}$ is the $j^{\text{th}}$ predictor's prediction for the $i^{\text{th}}$ instance.

  - The predictor's weight $\alpha_j$ is computed according to: $\alpha_j=\eta\log\frac{1-r_j}{r_j}$ where $\eta$ is the learning rate hyperparameter. The more accurate the predictor is, the higher its weight will be. If it is just guessing randomly, then its weight will be close to zero. However, if it is most often wrong (i.e., less accurate than random guessing), then its weight will be negative.

  - Next the instance weights are updated using: $w^{(i)} =w^{(i)} \exp(\alpha_j) $ if $\hat{y}_j^{(i)}\neq y^{(i)}$, otherwise stay unchanged. Then all the instance weights are normalized.

  - Finally, a new predictor is trained using the updated weights, and the whole process is repeated

- AdaBoost predictions: $\hat{y}(x)=\arg\max\limits_k\sum\limits_{j=1,\hat{y}_j(x)=k}^N\alpha_j$, where $N$ is the number of predictors.

- Scikit-Learn actually uses a multiclass version of AdaBoost called SAMME 16 (which stands for Stagewise Additive Modeling using a Multiclass Exponential loss function). When there are just two classes, SAMME is equivalent to AdaBoost. Moreover, if the predictors can estimate class probabilities (i.e., if they have a `predict_proba()` method), Scikit-Learn can use a variant of SAMME called SAMME.R (the R stands for “Real”), which relies on class probabilities rather than predictions and generally performs better.
  `ada_clf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)`

- If your AdaBoost ensemble is overfitting the training set, you can try reducing the number of estimators or more strongly regularizing the base estimator.



### Gradient Boosting

- Instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.
- It can make predictions on a new instance simply by adding up the predictions of all the predictors
- The `learning_rate` hyperparameter scales the contribution of each tree. If you set it to a low value, such as 0.1, you will need more trees in the ensemble to fit the training set, but the predictions will usually generalize better. This is a regularization technique called shrinkage.
- In order to find the optimal number of trees, you can use early stopping. A simple way to implement this is to use the `staged_predict()` method: it returns an iterator over the predictions made by the ensemble at each stage of training (with one tree, two trees, etc.).
- It is also possible to implement early stopping by actually stopping training early (instead of training a large number of trees first and then looking back to find the optimal number). You can do so by setting `warm_start=True`, which makes ScikitLearn keep existing trees when the `fit()` method is called, allowing incremental training.
- The GradientBoostingRegressor class also supports a `subsample` hyperparameter, which specifies the fraction of training instances to be used for training each tree. This trades a higher bias for a lower variance. It also speeds up training considerably. This technique is called **Stochastic Gradient Boosting**.
- It is possible to use Gradient Boosting with other cost functions. This is controlled by the loss hyperparameter.
- It is worth noting that an optimized implementation of Gradient Boosting is available in the popular python library **XGBoost**, which stands for Extreme Gradient Boosting.



## Stacking

- idea: instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation?
- Each predictor predicts a value, and the final predictor (called a **blender**, or a meta learner) takes these predictions as inputs and makes the final prediction
- To train the blender, a common approach is to use a hold-out set
  - First, the training set is split in two subsets. The first subset is used to train the predictors in the first layer
  - Next, the first layer predictors are used to make predictions on the second (held-out) set
  - We can create a new training set using these predicted values as input features, and keeping the target values. The blender is trained on this new training set, so it learns to predict the target value given the first layer’s predictions.



# Chapter 8 Dimensionality Reduction

- Reducing dimensionality does **lose some information** (just like compressing an image to JPEG can degrade its quality), so even though it will speed up training, it may also make your system perform slightly worse. So you should **first try to train your system with the original data** before considering using dimensionality reduction if training is too slow. In some cases, however, reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance (but in general it won’t; it will just speed up training).

- projection: In most real-world problems, training instances are not spread out uniformly across all dimensions. Many features are almost constant, while others are highly correlated. As a result, all training instances actually lie within (or close to) a much lower-dimensional subspace of the high-dimensional space.

- Put simply, a 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space. More generally, a d-dimensional manifold is a part of an n-dimensional space (where d < n) that locally resembles a d-dimensional hyperplane.

  

## PCA

- The direction of the principal components is not stable: if you perturb the training set slightly and run PCA again, some of the new PCs may point in the opposite direction of the original PCs. However, they will generally still lie on the same axes. In some cases, a pair of PCs may even rotate or swap, but the plane they define will generally remain the same.

- PCA assumes that the dataset is centered around the origin. As we will see, Scikit-Learn’s PCA classes take care of **centering the data** for you. However, if you implement PCA yourself (as in the preceding example), or if you use other libraries, don’t forget to center the data first.

### PCA and SVD

The goal of PCA is to project the dataset to a lower dimension space and at the same time preserve the variance as much as possible. Suppose $X =[x_1\quad x_2 \dots x_m] $ where $x_i$ is a column vector of dimension $n$. So the covariance of $X$ is $\frac1m XX^T$. 

Note that when we are considering a training dataset, it is expressed as $A=X^T$ instead of $X$. So the covariance matrix is $A^TA$.

SVD decomposes $A$ as $A=U\Sigma V^T $, and $V$ contains eigenvectors of $A^TA$. 



- `explained_variance_ratio_` variable indicates the proportion of the dataset’s variance that lies along the axis of each principal component.
- Instead of arbitrarily choosing the number of dimensions to reduce down to, it is generally preferable to choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%).



- Projecting the training set down to d dimensions: $X_{\text{d-proj}}=XW_d$, where $W_d=V[:d]$

- PCA inverse transformation, back to the original number of dimensions: $X_{\text{recovered}}=X_{\text{d-proj}}W_d^T$

### Randomized PCA

Scikit-Learn uses a **stochastic algorithm** called Randomized PCA that quickly finds an approximation of the first d principal components. Computational complexity $O(m\times d^2)+O(d^3)$, instead of $O(m\times n^2)+O(n^3)$ for the full SVD approach.

By default, `svd_solver` is actually set to "auto": Scikit-Learn automatically uses the randomized PCA algorithm if $m$ or $n$ is greater than 500 and d is less than 80% of $m$ or $n$, or else it uses the full SVD approach. If you want to force Scikit-Learn to use full SVD, you can set the `svd_solver` hyperparameter to "full".

### Incremental PCA

We can split the training set into mini-batches and feed an IPCA algorithm one mini-batch at a time.

use NumPy’s memmap class, which allows you to manipulate a large array stored in a binary file on disk as if it were entirely in memory. 

`X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))`

### Kernel PCA

Idea: a linear decision boundary in the high-dimensional feature space corresponds to a complex nonlinear decision boundary in the original space.

- As kPCA is an unsupervised learning algorithm, there is no obvious performance measure to help you select the best kernel and hyperparameter values. However, dimensionality reduction is often a preparation step for a supervised learning task (e.g., classification), so you can simply use grid search to select the kernel and hyperparameters that **lead to the best performance** on that task.
- Another approach, this time entirely unsupervised, is to select the kernel and hyperparameters that **yield the lowest reconstruction error**. Fortunately, it is possible to find a point in the original space that would map close to the reconstructed point. This is called the reconstruction pre-image.
- By default, fit_inverse_transform=False and KernelPCA has no `inverse_transform()` method. This method only gets created when you set `fit_inverse_transform=True`.



## LLE

- Locally Linear Embedding (LLE) is another very powerful nonlinear dimensionality reduction (NLDR) technique. It is a **Manifold Learning** technique that does not rely on projections like the previous algorithms. 
- In a nutshell, LLE works by first measuring how each training instance linearly relates to its closest neighbors (c.n.), and then looking for a low-dimensional representation of the training set where these **local relationships are best preserved** (more details shortly). This makes it particularly good at unrolling twisted manifolds, especially when there is not too much noise.

### Alg

- Step1: linearly modeling local relationships
  - first, for each training instance $x^{(i)}$ , the algorithm identifies its $k$ closest neighbors
  - Then tries to reconstruct $x^{(i)}$ as a linear function of these neighbors. Specifically, it finds the weights $w_{i,j}$ such that the squared distance between $x^{(i)}$ and $\sum_{j=1}^m w_{i,j}x^{(j)}$ is as small as possible, assuming $w_{i,j} = 0$ if $x^{(j)}$ is not one of the $k$ closest neighbors of $x^{(i)}$, $\sum_{j=1}^mw_{i,j}=1$. $\hat{W}=\arg\min_W\sum_{i=1}^m(x^{(i)}-\sum_{i=1}^mw_{i,j}x^{(j)})^2 $
- Step2: reducing dimensionality while preserving relationships
  - Now the second step is to map the training instances into a d-dimensional space (where $d < n$) while preserving these local relationships as much as possible. The weight found in step1 is $\hat{W}$. $z^{(i)}$ is the image of $x^{(i)}$  in this d-dimensional space
    $\hat{Z}=\arg\min_Z\sum_{i=1}^m(z^{(i)}-\sum_{j=1}^m\hat{w}_{i,j}z^{(j)})^2$

Computational complexity: $O(m\log(m)n\log(k))$ for finding the k nearest neighbors, $O(mnk^3)$ for optimizing the weights, and $O(dm^2)$ for constructing the low-dimensional representations. Unfortunately, the $m^2$ in the last term makes this algorithm scale poorly to very large datasets.

# Chapter 9 Unsupervised Learning Techniques

## Clustering

- it is the task of identifying similar instances and assigning them to clusters, i.e., groups of similar instances.
- Instead of assigning each instance to a single cluster, which is called **hard clustering**, it can be useful to just give each instance a score per cluster: this is called **soft clustering**.

## k-means

The computational complexity of the algorithm is generally **linear** with regards to the number of instances m, the number of clusters k and the number of dimensions n. However, this is only true when the data has a clustering structure. If it does not, then in the worst case scenario the complexity can increase exponentially with the number of instances. In practice, however, this rarely happens, and K-Means is generally one of the fastest clustering algorithms.

Unfortunately, although the algorithm is guaranteed to converge, it may not converge to the right solution

### Centroid Initialization Methods

- you can set the `init` hyperparameter to a NumPy array containing the list of centroids, and set `n_init` to 1.
- run the algorithm multiple times with different random initializations and keep the best solution. This is controlled by the `n_init` hyperparameter

The model’s **inertia**: this is the mean squared distance between each instance and its closest centroid.

K-Means++ initialization algorithm:

- Take one centroid $c^{(1)}$ , chosen uniformly at random from the dataset.
- Take a new centroid $c^{(i)}$, choosing an instance $x^{(i)}$ with probability: $D(x^{(i)})^2/\sum_{j=1}^mD(x^{(j)})^2$ where $D(x^{(i)})$ is the distance between $x^{(i)}$ and the closest centroid that was already chosen. This probability distribution ensures that instances further away from already chosen centroids are much more likely be selected as centroids.
- Repeat the previous step until all $k$ centroids have been chosen.

The KMeans class actually **uses this initialization method by default**.

### Accelerated K-Means and Mini-batch K-Means

Accelerated K-Means: it considerably accelerates the algorithm by avoiding many unnecessary distance calculations: this is achieved by exploiting the triangle inequality (i.e., the straight line is always the shortest) and by keeping track of lower and upper bounds for distances between instances and centroids. This is the algorithm used by **default** by the KMeans class.

Mini-batch K-Means: instead of using the full dataset at each iteration, the algorithm is capable of using mini-batches, moving the centroids just slightly at each iteration.

### Finding the Optimal Number of Clusters

- The inertia is not a good performance metric when trying to choose k since it keeps getting lower as we increase k.
- A coarse rule: elbow rule
- A more **precise approach** (but also more **computationally expensive**) is to use the **silhouette score**, which is the mean silhouette coefficient over all the instances. An instance’s silhouette coefficient is equal to $(b – a) / \max(a, b)$ where $a$ is the mean distance to the other instances in the same cluster (it is the mean intra-cluster distance), and $b$ is the mean nearest-cluster distance, that is the mean distance to the instances of the next closest cluster (defined as the one that minimizes $b$, excluding the instance’s own cluster). The silhouette coefficient can vary between -1 and +1.
- a coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary, and finally a coefficient close to -1 means that the instance may have been assigned to the wrong cluster.

### Limits of K-Means

- it is necessary to run the algorithm several times to avoid sub-optimal solutions
- you need to specify the number of clusters
- the performance depends on the data

It is important to **scale the input features** before you run K-Means, or else the clusters may be very stretched, and K-Means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things.

## Using Clustering for Semi-Supervised Learning

- First, let’s cluster the training set into 50 clusters, then for each cluster let’s find the image closest to the centroid.
  `k = 50` 
  `kmeans = KMeans(n_clusters=k)` 
  `X_digits_dist = kmeans.fit_transform(X_train)` 
  `representative_digit_idx = np.argmin(X_digits_dist, axis=0)` 
  `X_representative_digits = X_train[representative_digit_idx]`
- manually label the 50 representatives.
  ` y_representative_digits = np.array([4, 8, 0, 6, 8, 3, ..., 7, 6, 2, 3, 1, 1])`

Compared with using 50 randomly selected training instances, the accuracy rises from 82.7% accuracy to 92.4%.

Next step: **label propagation**
``y_train_propagated = np.empty(len(X_train), dtype=np.int32)` 
`for i in range(k):`
	`y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]`

We got a tiny little accuracy boost. Better than nothing, but not astounding. The problem is that we propagated each representative instance’s label to all the instances in the same cluster, **including the instances located close to the cluster boundaries**, which are more likely to be mislabeled.

Let’s see what happens if we only propagate the labels to the 20% of the instances that are closest to the centroids:

`percentile_closest = 20`
`X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]` 
`for i in range(k):`
   	`in_cluster = (kmeans.labels_ == i)` 
   	`cluster_dist = X_cluster_dist[in_cluster]` 
   	`cutoff_distance = np.percentile(cluster_dist, percentile_closest)` 
   	`above_cutoff = (X_cluster_dist > cutoff_distance)` 
   	`X_cluster_dist[in_cluster & above_cutoff] = -1`
`partially_propagated = (X_cluster_dist != -1)` 
`X_train_partially_propagated = X_train[partially_propagated]` 
`y_train_partially_propagated = y_train_propagated[partially_propagated]`

**active learning**: this is when a human expert interacts with the learning algorithm, providing labels when the algorithm needs them.

- uncertainty sampling:
  - The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled instances.
  - The instances for which the model is most uncertain (i.e., when its estimated probability is lowest) must be labeled by the expert.
  - Then you just iterate this process again and again, until the performance improvement stops being worth the labeling effort.

## DBSCAN

- For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s ε-neighborhood.
- If an instance has **at least min_samples instances in its ε-neighborhood** (including itself), then it is considered a core instance. In other words, core instances are those that are located in dense regions.
- All instances in the neighborhood of a core instance belong to the same cluster.
- This may include other core instances, therefore a long sequence of neighboring core instances forms a single cluster.
- Any instance that is not a core instance and does not have one in its neighborhood is considered an **anomaly**.

## Other Clustering Algorithms

- **Agglomerative clustering**: a hierarchy of clusters is built from the bottom up. At each iteration agglomerative clustering connects the nearest pair of clusters (starting with individual instances).
- **Birch**: this algorithm was designed specifically for very large datasets, and it can be faster than batch K-Means. It builds a tree structure during training containing just enough information to quickly assign each new instance to a cluster, without having to store all the instances in the tree: this allows it to use limited memory, while handle huge datasets
- **Mean-shift**: this algorithm starts by placing a circle centered on each instance, then for each circle it computes the mean of all the instances located within it, and it shifts the circle so that it is centered on the mean. Next, it iterates this mean-shift step until all the circles stop moving. Unfortunately, its computational complexity is $O(m^2)$, so it is not suited for large datasets.
- **Affinity propagation**: this algorithm uses a voting system, where instances vote for similar instances to be their representatives, and once the algorithm converges, each representative and its voters form a cluster. Unfortunately, its computational complexity is $O(m^2)$
- **Spectral clustering**: this algorithm takes a similarity matrix between the instances and creates a low-dimensional embedding from it, then it uses another clustering algorithm in this low-dimensional space. It does not scale well to large number of instances

## Gaussian Mixture

A **Gaussian mixture model** (GMM) is a probabilistic model that assumes that the instances were **generated from a mixture of several Gaussian distributions** whose parameters are unknown.

- For each instance, a cluster is picked randomly among $k$ clusters. The probability of choosing the $j^{\text{th}}$ cluster is defined by the cluster’s weight $\phi^{(j)}$. The index of the cluster chosen for the $i^{\text{th}}$ instance is noted $z^{(i)}$ .
- If $z^{(i)}=j$, meaning the $i^{\text{th}}$ instance has been assigned to the $j^{\text{th}}$ cluster, the location $x^{(i)}$ of this instance is sampled randomly from the Gaussian distribution with mean $\mu^{(j)}$ and covariance matrix $\Sigma^{(j)}$

To limit the range of shapes and orientations that the clusters can have, just set the `covariance_type` hyperparameter to one of the following values:

- "spherical": all clusters must be **spherical**, but they can have different diameters (i.e., different variances).
- "diag": clusters can take on any ellipsoidal shape of any size, but the ellipsoid’s axes must be parallel to the coordinate axes (i.e., the covariance matrices must be diagonal).
- "tied": all clusters must have **the same ellipsoidal shape, size and orientation** (i.e., all clusters share the same covariance matrix).
- By default, covariance_type is equal to "full", which means that each cluster can take on any shape, size and orientation (it has its own unconstrained covariance matrix)

### Anomaly Detection using Gaussian Mixtures

Anomaly detection (also called outlier detection) is the task of detecting instances that deviate strongly from the norm.

Identify the outliers using the 4th percentile lowest density as the threshold (i.e., approximately 4% of the instances will be flagged as anomalies):
`densities = gm.score_samples(X)`
`density_threshold = np.percentile(densities, 4)` 
`anomalies = X[densities < density_threshold]`

**novelty detection**: it differs from anomaly detection in that the algorithm is assumed to be trained on a “clean” dataset, uncontaminated by outliers, whereas anomaly detection does not make this assumption. Indeed, outlier detection is often precisely used to clean up a dataset.

Gaussian mixture models try to fit all the data, including the outliers, so if you have too many of them, this will bias the model’s view of “normality”: some outliers may wrongly be considered as normal. If this happens, you can try to fit the model once, use it to detect and remove the most extreme outliers, then fit the model again on the cleaned up dataset. Another approach is to use robust covariance estimation methods. 

### Selecting the Number of Clusters

For K-Means, use inertia or silhouette score. With Gaussian mixtures, it is not possible to use these metrics because they are not reliable when the clusters are not spherical or have different sizes.

Instead, you can try to find the model that **minimizes a theoretical information criterion** such as the **Bayesian information criterion** (BIC) or the **Akaike information criterion** (AIC)

- BIC = $log(m)p-2\log(\hat{L})$
- AIC = $2p-2\log(\hat{L})$
  $m$ is the number of instances, as always.
  $p$ is the number of parameters learned by the model.
  $L$ is the maximized value of the likelihood function of the model.

BIC and AIC often end up selecting the same model, but when they differ, the model selected by the BIC tends to be simpler (fewer parameters) than the one selected by the AIC, but it does not fit the data quite as well (this is especially true for larger datasets).

### Bayesian Gaussian Mixture Models

Rather than manually searching for the optimal number of clusters, it is possible to use instead the BayesianGaussianMixture class which is **capable of giving weights equal (or close) to zero to unnecessary clusters** . Just set the number of clusters `n_components` to a value that you have good reason to believe is greater than the optimal number of clusters (this assumes some minimal knowledge about the problem at hand), and the algorithm will **eliminate the unnecessary clusters automatically**.

In this model, the cluster parameters (including the weights, means and covariance matrices) are not treated as fixed model parameters anymore, but as **latent random variables**, like the cluster assignments.

Prior knowledge about the latent variables $z$ can be encoded in a probability distribution $p(z)$ called the **prior**. For example, we may have a prior belief that the clusters are likely to be few (low concentration), or conversely, that they are more likely to be plentiful (high concentration). This can be adjusted using the `weight_concentration_prior` hyperparameter.

 Bayes’ theorem tells us how to update the probability distribution over the latent variables after we observe some data X: $p(z|X)=\frac{p(X|z)p(z)}{p(X)}$. In a Gaussian mixture model (and many other problems), the denominator $p(X)$ is intractable, as it **requires integrating over all the possible values of $z$**.

There are several approaches to solving it. One of them is **variational inference**, which picks **a family of distributions** $q(z; \lambda)$ with its own variational parameters $\lambda$ (lambda), then it optimizes these parameters to make $q(z)$ a good approximation of $p(z|X)$ by minimizing the KL divergence from $q(z)$ to $p(z|X)$.

## Other Anomaly Detection and Novelty Detection Algorithms

- Fast-MCD (minimum covariance determinant), implemented by the EllipticEn velope class: this algorithm is useful for outlier detection, in particular to cleanup a dataset
- Isolation forest: this is an efficient algorithm for outlier detection, especially in high-dimensional datasets.
- Local outlier factor (LOF): this algorithm is also good for outlier detection.
- One-class SVM: this algorithm is better suited for novelty detection. The one-class SVM algorithm instead tries to separate the instances in high-dimensional space from the origin.
