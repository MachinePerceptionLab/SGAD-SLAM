#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>
#include <chrono> 
namespace fast_gicp {

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 10; 
  reg_name_ = "FastGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();
  knn_max_distance_ = 0.5;

  // k choices for point2plane when deciding the correspondence
  k_choice_p2p_ = 7;
  
  source_covs_.clear();  
  source_raw_covs_.clear();
  source_rotationsq_.clear();
  source_rotationsu_.clear();
  source_rotationsv_.clear();
  source_singular_.clear();
  source_scales_.clear();
  source_z_values_.clear();

  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();

  regularization_method_ = RegularizationMethod::NORMALIZED_ELLIPSE;
  search_source_.reset(new SearchMethodSource);
  search_target_.reset(new SearchMethodTarget);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::~FastGICP() {}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setNumThreads(int n) {
  num_threads_ = n;


#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setKNNMaxDistance(float k) {
  knn_max_distance_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setKChoiceP2P(int k) {
  k_choice_p2p_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);
  source_rotationsq_.swap(target_rotationsq_);
  source_scales_.swap(target_scales_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
  source_rotationsq_.clear();
  source_rotationsu_.clear();
  source_rotationsv_.clear();
  source_singular_.clear();
  source_scales_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  search_source_->setInputCloud(cloud);
  source_covs_.clear();
  source_raw_covs_.clear();
  source_rotationsq_.clear();
  source_rotationsu_.clear();
  source_rotationsv_.clear();
  source_singular_.clear();
  source_scales_.clear();
  // std::cout<<"set input source end"<<std::endl;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateSourceCovariance() {
	if (input_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	source_covs_.clear();
	source_rotationsq_.clear();
	source_scales_.clear();
	calculate_covariances(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateSourceCovs() {
	if (input_->size() == 0){
    std::cerr<<"no point cloud"<<std::endl;
    return;
  }
  source_raw_covs_.clear();
  calculate_source_covs(input_, *search_source_, source_covs_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateSourceCovarianceWithFilter() {
	if (input_->size() == 0){
    std::cerr<<"no point cloud"<<std::endl;
    return;
  }
  // source_raw_covs_.clear();
  source_covs_.clear();
  source_rotationsq_.clear();
  source_rotationsu_.clear();
  source_rotationsv_.clear();
  source_singular_.clear();
  source_scales_.clear();
  calculate_source_covariances_with_filter(input_, *search_source_, source_covs_, source_raw_covs_, source_rotationsu_, source_rotationsv_, source_singular_, source_rotationsq_, source_scales_, source_filter_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  search_target_->setInputCloud(cloud);
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
  // std::cout<<"set input target end"<<std::endl;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovariance() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithZ() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances_withz(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_z_values_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithFilter() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_target_covariances_with_filter(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_filter_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& input_rotationsu,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& input_rotationsv,
  const std::vector<float>& input_singular)
	{
		setSCovariances(input_rotationsq, input_scales, input_rotationsu, input_rotationsv, input_singular, source_covs_, source_rotationsq_, source_scales_, source_filter_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceZvalues(const std::vector<float>& input_z_values)
	{
		source_z_values_.clear();
    source_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    source_num_trackable_points_ = num_trackable_points;
		source_filter_.clear();
    source_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    target_num_trackable_points_ = num_trackable_points;
		target_filter_.clear();
    target_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetZvalues(const std::vector<float>& input_z_values)
	{
		target_z_values_.clear();
    target_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales)
	{
		setTCovariances(input_rotationsq, input_scales, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {
    throw std::invalid_argument("FastGICP: destination cloud cannot be identical to source or target");
  }
  // std::cout<<"source size: "<< input_->size() << std::endl;
  // std::cout<<"source cov size: "<< source_covs_.size() << std::endl;
  if (source_covs_.size() != input_->size()) {
    
    calculate_source_covariances_with_filter(input_, *search_source_, source_covs_, source_raw_covs_, source_rotationsu_, source_rotationsv_, source_singular_, source_rotationsq_, source_scales_, source_filter_);
  }
  if (target_covs_.size() != target_->size()) {
    // std::cout<<"compute target cov"<<std::endl;
    calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_);
  }
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
float FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::getFinalError() {
  return LsqRegistration<PointSource, PointTarget>::getFinalErrorlsq();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  point2plane_dist_.resize(input_->size());

  // std::vector<int> k_indices(1);
  // std::vector<float> k_sq_dists(1);
  const int n = k_choice_p2p_; // 7
  // std::cout << "--------------n--------------: " << n << std::endl;
  std::vector<int> k_indices(n);
  std::vector<float> k_sq_dists(n);
  

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();
    
  



    // /////////////// Point2Plane ////////
    // Get the n nearest neighbors
    search_target_->nearestKSearch(pt, n, k_indices, k_sq_dists);

    // get vector from source pt to target pt
    std::vector<Eigen::Vector3f> source_to_target(n);
    for (int j = 0; j < n; j++){
      source_to_target[j] = target_->at(k_indices[j]).getVector3fMap() - pt.getVector3fMap();
    }

    

    // get normal vector of target pt (from cov using svd)
    std::vector<Eigen::Vector3f> normals_target(n);
    for (int j = 0; j < n; j++){
      Eigen::Matrix4d cov = target_covs_[k_indices[j]];
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      // normals_target.col(j) = svd.matrixU().col(2).cast<float>();
      normals_target[j] = svd.matrixU().col(2).cast<float>();
    }


    // compute dot product of normal vector and source to target vector
    std::vector<float> dot_product(n);
    for (int j = 0; j < n; j++){
      dot_product[j] = normals_target[j].transpose() * source_to_target[j];
    }
    //get absolute value of dot product
    std::vector<float> abs_dot_product(n);
    for (int j = 0; j < n; j++){
      abs_dot_product[j] = std::abs(dot_product[j]);
    }
    // get min value of abs dot product
    float min_value = *std::min_element(abs_dot_product.begin(), abs_dot_product.end());
    int min_index = std::distance(abs_dot_product.begin(), std::min_element(abs_dot_product.begin(), abs_dot_product.end()));
 

    sq_distances_[i] = k_sq_dists[min_index];
    point2plane_dist_[i] = min_value;
    correspondences_[i] = k_sq_dists[min_index] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[min_index] : -1;
   
   
  

    


    



    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    if (RCR.determinant() == 0){
      // std::cout << "mahalanobis value will be NaN" << std::endl;
      mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
    else{
      mahalanobis_[i] = RCR.inverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
  }
}





template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences_py(const Matrix4& guess) {
  Eigen::Isometry3d trans = Eigen::Isometry3d(guess.template cast<double>());
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  point2plane_dist_.resize(input_->size());


  const int n = k_choice_p2p_; // 7
  // std::cout << "--------------n--------------: " << n << std::endl;
  std::vector<int> k_indices(n);
  std::vector<float> k_sq_dists(n);
  

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();
    
    


    // /////////////// Point2Plane ////////
    // Get the n nearest neighbors
    search_target_->nearestKSearch(pt, n, k_indices, k_sq_dists);

    // get vector from source pt to target pt
    std::vector<Eigen::Vector3f> source_to_target(n);
    for (int j = 0; j < n; j++){
      source_to_target[j] = target_->at(k_indices[j]).getVector3fMap() - pt.getVector3fMap();
    }

    

    // get normal vector of target pt (from cov using svd)
    std::vector<Eigen::Vector3f> normals_target(n);
    for (int j = 0; j < n; j++){
      Eigen::Matrix4d cov = target_covs_[k_indices[j]];
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      normals_target[j] = svd.matrixU().col(2).cast<float>();
    }


    // compute dot product of normal vector and source to target vector
    std::vector<float> dot_product(n);
    for (int j = 0; j < n; j++){
      dot_product[j] = normals_target[j].transpose() * source_to_target[j];
    }
    //get absolute value of dot product
    std::vector<float> abs_dot_product(n);
    for (int j = 0; j < n; j++){
      abs_dot_product[j] = std::abs(dot_product[j]);
    }
    // get min value of abs dot product
    float min_value = *std::min_element(abs_dot_product.begin(), abs_dot_product.end());
    int min_index = std::distance(abs_dot_product.begin(), std::min_element(abs_dot_product.begin(), abs_dot_product.end()));
 
   
    sq_distances_[i] = k_sq_dists[min_index];
    point2plane_dist_[i] = min_value;
    correspondences_[i] = k_sq_dists[min_index] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[min_index] : -1;
   
   



    



    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    if (RCR.determinant() == 0){
      // std::cout << "mahalanobis value will be NaN" << std::endl;
      mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
    else{
      mahalanobis_[i] = RCR.inverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
  }
}



template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

   
    // ORIGINAL
    sum_errors += error.transpose() * mahalanobis_[i] * error; 


    
    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    
    // ORIGINAL
    sum_errors += error.transpose() * mahalanobis_[i] * error; 

   
  }

  return sum_errors;
}




template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances_withz(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<float>& z_values
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();

    float z = std::max(1.,pow(z_values[i], 1.5)*2.);
    // std::cout<<z<<std::endl;
    scales[3*i+0] = (float)scale.x()/z;
    scales[3*i+1] = (float)scale.y()/z;
    scales[3*i+2] = (float)scale.z()/z;

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
          break;

      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }
  return true;
}


template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_source_covs(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& raw_covariances

  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(source_num_trackable_points_);


  // save covariances of trackable points
  raw_covariances.resize(cloud->size());


#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
    
    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }


    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    
    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    raw_covariances[i] = neighbors * neighbors.transpose() / k_correspondences_;
  }
  // std::cout<<"raw covs size after compute: "<<raw_covariances.size()<<std::endl;
  return true;
}
    


template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_source_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& raw_covariances,
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotationsu,
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotationsv,
  std::vector<float>& singular,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(source_num_trackable_points_);
  // pcl::copyPointCloud(*cloud, filter, *newCloud);

  // save covariances of trackable points
  covariances.resize(source_num_trackable_points_);
  if (raw_covariances.size() > 0){
  raw_covariances.resize(cloud->size());}
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());
  rotationsu.resize(cloud->size());
  rotationsv.resize(cloud->size());
  singular.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
  Eigen::Matrix4d cov;
  if (raw_covariances.size() > 0){
  // std::cout<<"using raw covs"<<std::endl;
  cov = raw_covariances[i];}
  else{
  // std::cout<<"computing raw covs"<<std::endl;
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
    
    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }


    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    
    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    cov = neighbors * neighbors.transpose() / k_correspondences_;}

    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());

    qfrommat.normalize();
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();

    rotationsu[i] = svd.matrixU();
    rotationsv[i] = svd.matrixV();
    singular[3*i+0] = (double)svd.singularValues()(0);
    singular[3*i+1] = (double)svd.singularValues()(1);
    singular[3*i+2] = (double)svd.singularValues()(2);

    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariance and xyz of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1, 1, 1e-3); 
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              values = svd.singularValues() / svd.singularValues()(1);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); 
        }
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);
  
      }
    }
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(newCloud);
  
  search_source_->setInputCloud(newCloud);
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_target_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(target_num_trackable_points_);


  // save covariances of trackable points
  covariances.resize(target_num_trackable_points_);
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();

    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariances of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1, 1, 1e-3); 
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              values = svd.singularValues() / svd.singularValues()(1);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); 
        }
        // std::cout << "regularization_method_ : " << values << std::endl;
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);
      }
    }
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(newCloud);
  search_target_->setInputCloud(newCloud);
 
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotationsu,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotationsv,
  const std::vector<float>& singular,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter) 
	{
	if(input_rotationsq.size()/4 != input_scales.size()/3){
		std::cerr << "size not match" <<std::endl;
		abort();
	}
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq;
	scales = input_scales;
  covariances.resize(source_num_trackable_points_);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<scales.size()/3; i++){
    Eigen::Vector3d singular_values = {(double)singular[3*i+0], (double)singular[3*i+1], (double)singular[3*i+2]};
    if (filter[i]!=0){
		switch (regularization_method_) {
		default:
		  std::cerr << "here must not be reached" << std::endl;
		  abort();
		case RegularizationMethod::PLANE:
		  singular_values = Eigen::Vector3d(1, 1, 1e-3); 
		  break;
		case RegularizationMethod::MIN_EIG:
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_MIN_EIG:
		  singular_values = singular_values / singular_values.maxCoeff();
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_ELLIPSE:
      if (singular_values(1) == 0){
          singular_values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
		  }
		  else{          
			  singular_values = singular_values / singular_values(1);
		  }
		  break;
    case RegularizationMethod::TEST:
      break;
		case RegularizationMethod::NONE:
		  // do nothing
		  break;
		case RegularizationMethod::FROBENIUS:
		  std::cerr<< "should be implemented"<< std::endl;
		  abort();
	      }

           Eigen::Quaterniond q;
            q.x() = (double)rotationsq[4*i+0];
            q.y() = (double)rotationsq[4*i+1];
            q.z() = (double)rotationsq[4*i+2];
            q.w() = (double)rotationsq[4*i+3];
           


	      q = q.normalized();
	      covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = rotationsu[i] * singular_values.asDiagonal() * rotationsv[i].transpose();
     
      }
  }
  
}



template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d,
  Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  
  std::vector<float>& rotationsq,
  std::vector<float>& scales) 
	{
	if(input_rotationsq.size()/4 != input_scales.size()/3){
		std::cerr << "size not match" <<std::endl;
		abort();
	}
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq;
	scales = input_scales;
	covariances.resize(input_scales.size()/3);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<scales.size()/3; i++){
		Eigen::Vector3d singular_values = { (double)scales[3*i+0]*scales[3*i+0], 
							(double)scales[3*i+1]*scales[3*i+1], 
							(double)scales[3*i+2]*scales[3*i+2] };
		switch (regularization_method_) {
		default:
		  std::cerr << "here must not be reached" << std::endl;
		  abort();
		case RegularizationMethod::PLANE:
		  singular_values = Eigen::Vector3d(1, 1, 1e-3); 
		  break;
		case RegularizationMethod::MIN_EIG:
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_MIN_EIG:
		  singular_values = singular_values / singular_values.maxCoeff();
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_ELLIPSE:
		  if (singular_values(1) < 1e-3){
		  	singular_values = Eigen::Vector3d(1e-3, 1e-3, 1e-3);
		  }
		  else{          
			  singular_values = singular_values / singular_values(1);
		  }
		  break;
    case RegularizationMethod::TEST:
      break;
		case RegularizationMethod::NONE:
		  // do nothing
		  break;
		case RegularizationMethod::FROBENIUS:
		  std::cerr<< "should be implemented"<< std::endl;
		  abort();
	      }
	      Eigen::Quaterniond q( (double)rotationsq[4*i+0], 
	      				(double)rotationsq[4*i+1], 
	      				(double)rotationsq[4*i+2], 
	      				(double)rotationsq[4*i+3]);
        
	      q = q.normalized();
	      covariances[i].setZero();
	      covariances[i].template block<3, 3>(0, 0) = q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
  }
  
}

}  // namespace fast_gicp

#endif
