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
        rho_(rho),
        camera_params_multipler_(camera_params.size(), 0.0),
        pose_params_multipler_(pose_params.size(), 0.0),
        structure_params_multipler_(structure_params.size(), 0.0),
        camera_consensus_params_(camera_params.size(), 0.0),
        pose_consensus_params_(pose_params.size(), 0.0),
        structure_consensus_params_(structure_params.size(), 0.0)
         {}
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
    Eigen::VectorXd x =
        Eigen::Map<Eigen::VectorXd>(&camera_params_[0], camera_params_.size());
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(
        &camera_params_multipler_[0], camera_params_.size());
    Eigen::VectorXd res = (x - 1.0 / rho_ * y);
    std::vector<double> res_vector;
    for (int i = 0; i < camera_params_.size(); i++) {
      res_vector.push_back(res(i));
    }
    return res_vector;
  };

  std::vector<double> getLocalPoseParams() override {
    return ZUpdate(pose_params_, pose_params_multipler_, rho_);
  };

  std::vector<double> getLocalStructureParams() override {
    return ZUpdate(structure_params_, structure_params_multipler_, rho_);
  };

  void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_consensus_params) override {
    assert(camera_consensus_params.size() == camera_consensus_params_.size());
    camera_consensus_params_ = camera_consensus_params;
    std::vector<double> delta_y = YUpdate(camera_params_, camera_consensus_params_, rho_);
    for (int i = 0; i < camera_params_multipler_.size(); i++) {
      camera_params_multipler_[i] += delta_y[i];
    }
  }

  virtual void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_consensus_params) override {
    assert(pose_consensus_params.size() == pose_consensus_params_.size());
    pose_consensus_params_ = pose_consensus_params;
    std::vector<double> delta_y = YUpdate(pose_params_, pose_consensus_params_, rho_);
    for (int i = 0; i < pose_params_multipler_.size(); i++) {
      pose_params_multipler_[i] += delta_y[i];
    }
  }

  virtual void setUpdateStructureParams(
      const std::vector<double>& structure_consensus_params) override {
    assert(structure_consensus_params.size() ==
           structure_consensus_params_.size());
    structure_consensus_params_ = structure_consensus_params;
    std::vector<double> delta_y = YUpdate(structure_params_, structure_consensus_params_, rho_);
    for(int i = 0; i < structure_params_multipler_.size(); i++) {
      structure_params_multipler_[i] += delta_y[i];
    }
  }

 private:
  std::vector<double> ZUpdate(std::vector<double> origin_params,
                              std::vector<double> origin_params_multipler,
                              double rho) {
    Eigen::VectorXd x =
        Eigen::Map<Eigen::VectorXd>(&origin_params[0], origin_params.size());
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(
        &origin_params_multipler[0], origin_params_multipler.size());
    Eigen::VectorXd res = (x - 1.0 / rho_ * y);
    std::vector<double> res_vector;
    for (int i = 0; i < origin_params.size(); i++) {
      res_vector.push_back(res(i));
    }
    return res_vector;
  }
  std::vector<double> YUpdate(std::vector<double>& origin_params, std::vector<double>& origin_consusens_params, double rho) {
    std::vector<double> res;
    for (int i = 0; i < origin_params.size();i++) {
      res.push_back(rho * (origin_params[i] - origin_consusens_params[i]));
    }
    return res;

  }
};

}  // namespace sfm

}  // namespace openMVG
#endif  // SFM_DATA_BA_PROCESSOR_HPP_