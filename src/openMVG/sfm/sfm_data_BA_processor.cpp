#include "sfm_data_BA_processor.hpp"
#include "ceres/problem.h"

ceres::CostFunction* IntrinsicsToCostFunction(
    openMVG::cameras::EINTRINSIC intrinsic, const openMVG::Vec2& observation,
    const double weight) {
  switch (intrinsic) {
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic::Create(
          observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL1:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL3:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_BROWN:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_FISHEYE:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::
          Create(observation, weight);
    //case openMVG::cameras::EINTRINSIC::CAMERA_SPHERICAL:
    //  return openMVG::sfm::ResidualErrorFunctor_Intrinsic_Spherical::Create(
    //     intrinsic, observation, weight);
    default:
      return {};
  }
}

namespace openMVG {
namespace sfm {


void CPUProcessor::OptimizeParameters() {
    ceres::CostFunction* intrins_cost_function = IntrinsicsToCostFunction(camera_type_, Eigen::Map<const Eigen::Vector2d>(&observations_[0]), structure_weight_);
    ceres::Problem problem;

    problem.AddParameterBlock(&camera_params_[0], camera_params_.size());
    problem.AddParameterBlock(&pose_params_[0], pose_params_.size());
    problem.AddParameterBlock(&structure_params_[0], structure_params_.size());

    problem.AddResidualBlock(intrins_cost_function, nullptr, &camera_params_[0], &pose_params_[0], &structure_params_[0]);
    ceres::Solver::Options options; 
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}
}  // namespace sfm
}  // namespace openMVG
