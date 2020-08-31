#include "sfm_data_BA_processor.hpp"

#include <memory>

#include "ceres/rotation.h"
#include "testing/testing.h"

TEST(BUNDLE_ADJUSTMENT_ADMM, TEST) { EXPECT_TRUE(true); }

TEST(BUNDLE_ADJUSTMENT_ADMM, Optimize) {
  std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic =
      std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024,
                                                            960, 540);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  std::vector<double> pose_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  ceres::RotationMatrixToAngleAxis(rotation.data(), &pose_vector[0]);
  std::vector<double> X = {1.1, 2.2, 3.8};
  // u = 0.25 * 1024 + 960 = 1216
  // v = 0.50 * 1024 + 540 = 1052
  std::vector<double> x = {1216, 1052};
  openMVG::sfm::Optimize_Options options;
  options.intrinsics_opt = openMVG::cameras::Intrinsic_Parameter_Type::NONE;
  options.extrinsics_opt = openMVG::sfm::Extrinsic_Parameter_Type::NONE;
  options.structure_opt = openMVG::sfm::Structure_Parameter_Type::ADJUST_ALL;
  std::shared_ptr<openMVG::sfm::Processor> processor =
      std::make_shared<openMVG::sfm::CPUProcessor>(
          intrinsic.get(), intrinsic->getParams(), pose_vector, X, x, options);

  for (int i  = 0; i < 1024; i++) {
    processor->XOptimization();
    processor->ZOptimization();
    std::vector<double> temp_structure = processor->getLocalStructureParams();
    temp_structure[0] /= temp_structure[2];
    temp_structure[1] /= temp_structure[2];
    temp_structure[2] = 1.0;
    processor->setUpdateStructureParams(temp_structure);
    processor->YUpdate();
  }
  std::vector<double> result = processor->getLocalStructureParams();
  result[0] /= result[2];
  result[1] /= result[2];
  EXPECT_NEAR(0.25, result[0], 1e-5);
  EXPECT_NEAR(0.50, result[1], 1e-5);
}

std::vector<std::shared_ptr<openMVG::sfm::Processor>> CreateScene() {
  std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic1 =
      std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024,
                                                            960, 540);

  std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic2 =
      std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024,
                                                            960, 540);

  openMVG::geometry::Pose3 pose1 = openMVG::geometry::Pose3();
  std::vector<double> pose_vector1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  ceres::RotationMatrixToAngleAxis(pose1.rotation().data(),
                                   pose_vector1.data());

  openMVG::geometry::Pose3 pose2 = openMVG::geometry::Pose3();
  std::vector<double> pose_vector2 = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  ceres::RotationMatrixToAngleAxis(pose2.rotation().data(),
                                   pose_vector2.data());
  // should be 1, 2, 4
  std::vector<double> X1 = {0.0, 0.0, 0.1};
  std::vector<double> X2 = {0.0, 0.0, 0.1};
  std::vector<double> x1 = {-704, -28};
  std::vector<double> x2 = {0, -28};

  openMVG::sfm::Optimize_Options options;
  options.intrinsics_opt = openMVG::cameras::Intrinsic_Parameter_Type::NONE;
  options.extrinsics_opt = openMVG::sfm::Extrinsic_Parameter_Type::NONE;
  options.structure_opt = openMVG::sfm::Structure_Parameter_Type::ADJUST_ALL;
  std::shared_ptr<openMVG::sfm::Processor> processor =
      std::make_shared<openMVG::sfm::CPUProcessor>(
          intrinsic1.get(), intrinsic1->getParams(), pose_vector1, X1, x1,
          options);

  std::shared_ptr<openMVG::sfm::Processor> processor2 =
      std::make_shared<openMVG::sfm::CPUProcessor>(
          intrinsic2.get(), intrinsic2->getParams(), pose_vector2, X2, x2,
          options);

  return {processor, processor2};
}

/*
TEST(BUNDLE_ADJUSTMENT_ADMM, Optimize_Scene) {
  std::vector<std::shared_ptr<openMVG::sfm::Processor>> processors =
      CreateScene();
  std::vector<double> global_camera_intrinsic;
  std::vector<double> global_point_parameter;
  for (int i = 0; i < 1024; i++) {
    for (auto processor : processors) {
      processor->OptimizeParameters();
      std::vector<double> camera_intrinsic = processor->getLocalCameraParams();

      if (global_camera_intrinsic.size() < camera_intrinsic.size()) {
        std::copy(camera_intrinsic.begin(), camera_intrinsic.end(),
                  std::back_inserter(global_camera_intrinsic));
      } else {
        for (int i = 0; i < camera_intrinsic.size(); i++) {
          global_camera_intrinsic[i] += camera_intrinsic[i];
          global_camera_intrinsic[i] /= 2.0;
        }
      }

      processor->setUpdatePoseParamsConsusence(processor->getLocalPoseParams());
      std::vector<double> structure = processor->getLocalStructureParams();

      if (global_point_parameter.size() < structure.size()) {
        std::copy(structure.begin(), structure.end(),
                  std::back_inserter(global_point_parameter));
      } else {
        for (int i = 0; i < structure.size(); i++) {
          global_point_parameter[i] += structure[i];
          global_point_parameter[i] /= 2.0;
        }
      }
    }
  }
  EXPECT_NEAR(1.0, global_point_parameter[0], 1e-5);
  EXPECT_NEAR(2.0, global_point_parameter[1], 1e-5);
  EXPECT_NEAR(4.0, global_point_parameter[2], 1e-5);
}
*/
struct MinimumProblem {
 public:
  MinimumProblem(double x1, double y1, double x2, double y2) {
    ob1_x = x1;
    ob1_y = y1;
    ob2_x = x2;
    ob2_y = y2;
  }
  template <class T>
  bool operator()(const T* X, T* residual) const {
    Eigen::Matrix<T, 3, 1> eigen_x;
    eigen_x << T(X[0]), T(X[1]), T(X[2]);
    Eigen::Matrix<T, 3, 3> R1, R2;
    R1 << T(1.0), T(0.0), T(0.0), T(0.0), T(1.0), T(0.0), T(0.0), T(0.0),
        T(1.0);

    R2 << T(1.0), T(0.0), T(0.0), T(0.0), T(1.0), T(0.0), T(0.0), T(0.0),
        T(1.0);
    Eigen::Matrix<T, 3, 1> t1, t2;
    t1 << T(0.0), T(0.0), T(0.0);
    t2 << T(1.0), T(0.0), T(0.0);
    T focal(1024), cx(960), cy(540);
    T u1 = X[0] / X[2];
    T v1 = X[1] / X[2];
    T u2 = (X[0] - T(1.0)) / X[2];
    T v2 = X[1] / X[2];

    residual[0] = focal * u1 - cx - T(ob1_x);
    residual[1] = focal * v1 - cy - T(ob1_y);
    residual[2] = focal * u2 - cx - T(ob2_x);
    residual[3] = focal * v2 - cy - T(ob2_y);
    return true;
  }

 private:
  double ob1_x, ob1_y, ob2_x, ob2_y;
};

TEST(BUNDLE_TRIANGLE, MinimumProblem) {
  ceres::Problem problem;
  ceres::CostFunction* cost =
      new ceres::AutoDiffCostFunction<MinimumProblem, 4, 3>(
          new MinimumProblem(-704, -28, -960, -28));
  std::vector<double> X = {1.0, 0.0, 1.0};
  problem.AddResidualBlock(cost, nullptr, &X[0]);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  EXPECT_NEAR(1.0, X[0], 1e-5);
  EXPECT_NEAR(2.0, X[1], 1e-5);
  EXPECT_NEAR(4.0, X[2], 1e-5);
}
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}