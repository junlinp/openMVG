#ifndef SFM_DATA_BA_PROCESSOR_HPP_
#define SFM_DATA_BA_PROCESSOR_HPP_
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Spherical.hpp"
#include "sfm_data_BA.hpp"
#include "sfm_data_BA_ceres_camera_functor.hpp"
namespace openMVG {
namespace sfm {
class Processor {
 public:
  Processor(openMVG::cameras::EINTRINSIC camera_type,
            const std::vector<double>& camera_params,
            const std::vector<double>& pose_params,
            const std::vector<double>& structure_params,
            const std::vector<double>& observation,
            const Optimize_Options& options,
            const double structure_weight = 0.0, const double rho = 1.0)
      : camera_type_(camera_type),
        camera_params_(camera_params),
        pose_params_(pose_params),
        structure_params_(structure_params),
        observations_(observation),
        options_(options),
        structure_weight_(structure_weight),
        rho_(rho) {}
  virtual ~Processor() {}
  virtual void OptimizeParameters() = 0;
  // TODO: junlinp@qq.com
  // return params + 1.0 / rho * params_multipler
  virtual std::vector<double> getLocalCameraParams() = 0;
  virtual std::vector<double> getLocalPoseParams() = 0;
  virtual std::vector<double> getLocalStructureParams() = 0;

  virtual void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_params) {}
  virtual void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_params) {}
  virtual void setUpdateStructureParams(
      const std::vector<double>& structure_params) {}

 protected:
  openMVG::cameras::EINTRINSIC camera_type_;
  std::vector<double> camera_params_;
  std::vector<double> pose_params_;
  std::vector<double> structure_params_;
  std::vector<double> observations_;
  std::vector<double> camera_params_multipler_;
  std::vector<double> pose_params_multipler_;
  std::vector<double> structure_params_multipler_;
  std::vector<double> camera_consensus_params_;  // z
  std::vector<double> pose_consensus_params_;
  std::vector<double> structure_consensus_params_;

  Optimize_Options options_;
  double structure_weight_;
  double rho_;
};

class CPUProcessor : public Processor {
 public:
  CPUProcessor(openMVG::cameras::EINTRINSIC camera_type,
               const std::vector<double>& camera_params,
               const std::vector<double>& pose_params,
               const std::vector<double>& structure_params,
               const std::vector<double>& observation,
               const Optimize_Options& options,
               const double structure_weight = 0.0, const double rho = 1.0)
      : Processor(camera_type, camera_params, pose_params, structure_params,
                  observation, options, structure_weight, rho) {}

  void OptimizeParameters() override;

  std::vector<double> getLocalCameraParams() override {
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(&camera_params_[0], camera_params_.size());
    Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd>(&camera_params_[0], camera_params_.size());
    return camera_params_multipler_;
  };

  std::vector<double> getLocalPoseParams() override {
    return pose_params_multipler_;
  };

  std::vector<double> getLocalStructureParams() override {
    return structure_params_multipler_;
  };

  void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_consensus_params) override {
    assert(camera_consensus_params.size() == camera_consensus_params_.size());
    camera_consensus_params_ = camera_consensus_params;
  }

  virtual void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_consensus_params) override {
    assert(pose_consensus_params.size() == pose_consensus_params_.size());
    pose_consensus_params_ = pose_consensus_params;
  }

  virtual void setUpdateStructureParams(
      const std::vector<double>& structure_consensus_params) {
    assert(structure_consensus_params.size() ==
           structure_consensus_params_.size());
    structure_consensus_params_ = structure_consensus_params;
  }
};

}  // namespace sfm

}  // namespace openMVG
#endif  // SFM_DATA_BA_PROCESSOR_HPP_